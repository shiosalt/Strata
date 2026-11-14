"""serve/test_nemotron_template.py - NVIDIA-Nemotron-3.5-Super-VL through the server (a scripted engine, no GPU): the tokenizer and chat
template taken from the GGUF's metadata shard (tools/strata_tokenizer.py, the tekken rule), the prompt the template renders, thinking
and the XML tool calls (<tool_call><function=..><parameter=..>) coming back as the API's reasoning and tool_calls.

Needs the real metadata shard (about 10 MB, no weights are read): STRATA_NEMOTRON_GGUF, or the path below; skipped without it.

    python -m pytest serve/test_nemotron_template.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SHARD = os.environ.get("STRATA_NEMOTRON_GGUF", "")   # the first shard (...-00001-of-00003.gguf); without it the tests are skipped


@unittest.skipUnless(SHARD and os.path.exists(SHARD), "STRATA_NEMOTRON_GGUF does not name the Nemotron GGUF's first shard")
class NemotronTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        cls.dir = tempfile.TemporaryDirectory()
        cfg = ST.extract(SHARD, cls.dir.name)
        tdir = Path(cls.dir.name) / "tokenizer"
        cls.cfg = cfg
        vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tdir / "token_type.json").read_text())
        tcfg = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
        cls.tok = ST.Tokenizer(tokens, merges, types, tcfg["pre"], tcfg.get("special_ids"))   # as serve/server.py builds it
        cls.template = ChatTemplate(tdir / "chat_template.jinja")

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def service(self, script):
        from serve.server import MockEngine, Service, serve
        svc = Service(MockEngine(self.tok, script, max_context=32768), self.tok, self.template, model_name="nemotron-3.5-super-vl-120b-a12b")
        svc.model_arch = "nemotron_h_moe"
        httpd = serve(svc, port=0)
        self.addCleanup(lambda: (httpd.shutdown(), httpd.server_close()))
        return svc, f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(self, base, body):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def test_tokenizer_folder(self):
        self.assertEqual(self.cfg["pre"], "tekken")
        self.assertEqual(self.cfg["vocab_size"], 131072)
        self.assertEqual(self.tok.encode("<|im_end|>", parse_special=True), [11])

    def test_prompt_ends_with_the_assistant_and_open_thinking(self):
        text = self.template.render([{"role": "user", "content": "Hello"}], add_generation_prompt=True)
        self.assertTrue(text.startswith("<|im_start|>system\n"), text[:80])
        self.assertTrue(text.endswith("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"), text[-120:])
        off = self.template.render([{"role": "user", "content": "Hello"}], add_generation_prompt=True, enable_thinking=False)
        self.assertTrue(off.endswith("<|im_start|>assistant\n<think></think>"), off[-80:])

    def test_answer_with_thinking(self):
        svc, base = self.service("I should greet.\n</think>\n\nHello there!")
        d = self.post(base, {"model": "m", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64})
        msg = d["choices"][0]["message"]
        self.assertEqual((msg.get("content") or "").strip(), "Hello there!")
        self.assertIn("I should greet.", msg.get("reasoning_content") or msg.get("reasoning") or "")
        self.assertEqual(d["choices"][0]["finish_reason"], "stop")
        # the engine was sent the template's prompt, tokenized with the tekken rule (the stop token is <|im_end|> = 11)
        self.assertEqual(svc.engine.last_prompt[-3:], self.tok.encode("assistant\n<think>\n", parse_special=True)[-3:])
        self.assertIn(11, svc.stop_ids)

    def test_thinking_that_ends_right_after_a_quote(self):
        """Nemotron ends its thinking with no line break, here right after a quoted draft (seen on the real model, 2026-11-10): the
        answer is the content, not more reasoning."""
        _, base = self.service('Draft: "It is a white letter a."</think>This picture shows a white letter **a**.')
        d = self.post(base, {"model": "m", "messages": [{"role": "user", "content": "What is in it?"}], "max_tokens": 64})
        msg = d["choices"][0]["message"]
        self.assertEqual((msg.get("content") or "").strip(), "This picture shows a white letter **a**.")
        self.assertNotIn("</think>", msg.get("reasoning_content") or msg.get("reasoning") or "")

    def test_tool_call(self):
        call = ("Let me look.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
                "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
        _, base = self.service(call)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "the weather",
                                                    "parameters": {"type": "object", "properties": {
                                                        "city": {"type": "string"}, "days": {"type": "integer"}},
                                                        "required": ["city"]}}}]
        d = self.post(base, {"model": "m", "messages": [{"role": "user", "content": "Weather in Tokyo?"}], "tools": tools, "max_tokens": 128})
        ch = d["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls", ch)
        tc = ch["message"]["tool_calls"]
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc[0]["function"]["arguments"]), {"city": "Tokyo", "days": 3})

    def test_tool_result_round_trip_renders(self):
        msgs = [{"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "get_weather", "arguments": {"city": "Tokyo"}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        text = self.template.render(msgs, add_generation_prompt=True)
        self.assertIn("<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>", text)
        self.assertIn("<tool_response>\nsunny\n</tool_response>", text)

    def test_pictures_become_img_runs(self):
        """Each picture: the template's <image> becomes <img>, one <image> per image token, </img> (llama.cpp mtmd's form for the
        nemotron_omni projector); "<image>" typed in a message stays text and does not take a picture's place."""
        from serve.server import MockEngine, Service

        class FakeVision:
            def __init__(self, d):
                self.dir = Path(d)
                self.rows = self.dir / "0123456789abcdef0123.sve"
                self.rows.write_bytes(b"rows")

            def encode(self, source):
                return self.rows, 6

        img, beg, end = (self.tok.encode(s, parse_special=True) for s in ("<image>", "<img>", "</img>"))
        self.assertEqual((img, beg, end), ([18], [19], [20]))
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(self.tok, "ok", max_context=32768), self.tok, self.template, vision=FakeVision(d))
            svc.model_arch = "nemotron_h_moe"
            msgs = [{"role": "user", "content": [{"type": "text", "text": "the tag <image> is text here; first"},
                                                 {"type": "image", "source": "a.png"},
                                                 {"type": "text", "text": "then"},
                                                 {"type": "image", "source": "b.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            self.assertEqual(ids.count(18), 12, "two pictures of 6 image tokens, and no other <image>")
            self.assertEqual(ids.count(19), 2)
            self.assertEqual(ids.count(20), 2)
            at = ids.index(19)
            self.assertEqual(ids[at:at + 8], [19] + [18] * 6 + [20])
            self.assertIn("the tag <image> is text here", self.tok.decode(ids))
            self.assertEqual(svc.embeddings.keys, ["0123456789abcdef", "0123456789abcdef"])
            svc.embeddings.path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
