"""Generation backends — the half of the loop that proposes new programs.

`AnthropicGenerator` calls Claude and needs a credential. `ManualGenerator`
writes each prompt to a file and reads a response back from a sibling file, so
the same loop can be driven by hand (or by an agent in the loop) with no
credential at all.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the single python block out of a model response."""
    blocks = CODE_BLOCK.findall(text)
    if not blocks:
        raise ValueError("response contained no fenced code block")
    # If the model emitted several, the longest is the program rather than a snippet.
    return max(blocks, key=len).strip()


class Generator(Protocol):
    def generate(self, system: str, prompt: str, tag: str) -> str:
        """Return the proposed replacement block for one child."""


class AnthropicGenerator:
    """Calls Claude through the Anthropic SDK.

    Streams because `max_tokens` is large enough that a non-streaming request
    risks an HTTP timeout. Note there is no `temperature` knob: it is rejected on
    Claude Opus 5 and Sonnet 5, so per-child variation comes from the resampled
    prompt instead.
    """

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 32000,
                 effort: str = "high"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._use_fallbacks = True

    def _request(self, system: str, prompt: str, with_fallbacks: bool):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if with_fallbacks:
            # Reruns the request on a fallback model if a safety classifier
            # declines it, so one refusal cannot stall the generation.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
            with self.client.beta.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        with self.client.messages.stream(**kwargs) as stream:
            return stream.get_final_message()

    def generate(self, system: str, prompt: str, tag: str) -> str:
        anthropic = self._anthropic
        try:
            try:
                message = self._request(system, prompt, self._use_fallbacks)
            except anthropic.BadRequestError:
                if not self._use_fallbacks:
                    raise
                # The account may not have the server-side-fallback beta enabled;
                # drop it for this and every later call rather than dying.
                self._use_fallbacks = False
                message = self._request(system, prompt, False)
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                "no valid Anthropic credential — set ANTHROPIC_API_KEY, or use "
                "--generator manual"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise RuntimeError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(f"could not reach the API: {exc}") from exc

        if message.stop_reason == "refusal":
            detail = getattr(message.stop_details, "explanation", "no explanation")
            raise RuntimeError(f"model declined to answer: {detail}")

        text = "".join(b.text for b in message.content if b.type == "text")
        return extract_code(text)


class ManualGenerator:
    """File-backed generation, for running without a credential.

    `generate` writes `<tag>.prompt.md` and looks for `<tag>.response.md`. If the
    response is not there yet it raises `PendingResponse`, which the driver
    reports so the prompts can be answered and the step rerun.
    """

    class PendingResponse(Exception):
        def __init__(self, paths: list[str]):
            self.paths = paths
            super().__init__(f"awaiting {len(paths)} response file(s)")

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def generate(self, system: str, prompt: str, tag: str) -> str:
        prompt_path = os.path.join(self.directory, f"{tag}.prompt.md")
        response_path = os.path.join(self.directory, f"{tag}.response.md")

        with open(prompt_path, "w") as fh:
            fh.write(f"<!-- system -->\n{system}\n\n<!-- user -->\n{prompt}")

        if not os.path.exists(response_path):
            raise self.PendingResponse([response_path])

        with open(response_path) as fh:
            return extract_code(fh.read())
