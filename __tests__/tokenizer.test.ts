import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import reference from "./reference.fixture.json";
import { WordPieceTokenizer } from "../src/tokenizer";

const vocab = readFileSync(
  new URL("../model/vocab.txt", import.meta.url),
  "utf8",
);
const tokenizer = new WordPieceTokenizer(vocab);

describe("WordPieceTokenizer", () => {
  it("loads the gate vocabulary with BERT special ids", () => {
    expect(tokenizer.vocabSize).toBe(23_910);
    expect(tokenizer.clsId).toBe(2);
    expect(tokenizer.sepId).toBe(3);
    expect(tokenizer.unkId).toBe(1);
  });

  it("rejects a vocabulary without the BERT special tokens", () => {
    expect(() => new WordPieceTokenizer("hello\nworld\n")).toThrow(
      "vocab.txt has no [UNK] entry; not a BERT vocabulary",
    );
  });

  expect(reference.tokenizer.length).toBeGreaterThan(10);
  for (const { text, ids } of reference.tokenizer) {
    it(`matches transformers.BertTokenizer for ${JSON.stringify(text.slice(0, 48))}`, () => {
      expect(tokenizer.encode(text)).toEqual(ids);
    });
  }

  it("encodeWithOffsets yields the same ids as encode", () => {
    for (const { text } of reference.tokenizer) expect(tokenizer.encodeWithOffsets(text).ids).toEqual(tokenizer.encode(text));
  });
});
