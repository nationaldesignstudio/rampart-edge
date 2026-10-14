import { beforeAll, describe, expect, it } from "vitest";

import { loadLocalGate } from "./node-gate";
import { compileAllowlist } from "../src/allowlist";
import { createValidatorGate, type Gate } from "../src/gate";

const MODEL = new URL("../model/", import.meta.url);

describe("allowlist", () => {
  let gate: Gate;
  let strict: Gate;
  beforeAll(async () => {
    gate = await loadLocalGate();
    strict = await loadLocalGate(undefined, MODEL, { builtIn: false });
  });

  const publicFigures = [
    "Write a story about Alexander Hamilton",
    "Write a story about John Marshall",
    "Write a story about Andrew Jackson",
  ];
  it.each(publicFigures)("allows built-in public figures the model blocks: %s", async (text) => {
    expect((await strict.check(text)).verdict).toBe("BLOCK");
    const decision = await gate.check(text);
    expect(decision).toMatchObject({ verdict: "ALLOW", basis: "allowlist", spans: [] });
  });

  it("allows the SSA hotline and the White House address from the built-in lists", async () => {
    const phone = "Call the SSA at 1-800-772-1213";
    const address = "Send it to 1600 Pennsylvania Avenue NW, Washington, DC 20500";
    expect((await strict.check(phone)).verdict).toBe("BLOCK");
    expect((await strict.check(address)).verdict).toBe("BLOCK");
    expect(await gate.check(phone)).toMatchObject({ verdict: "ALLOW", spans: [] });
    expect(await gate.check(address)).toMatchObject({ verdict: "ALLOW", basis: "allowlist", spans: [] });
  });

  it("allows a named .gov email the model blocks", async () => {
    const text = "My work email is jane.doe@va.gov, can you send the form there?";
    expect(await gate.check(text)).toMatchObject({ verdict: "ALLOW", basis: "allowlist", spans: [] });
  });

  it("still blocks a private name next to an allowlisted one, and returns only the private span", async () => {
    const text = "Write a story about Alexander Hamilton and my landlord Marcus Delgado";
    const decision = await gate.check(text);
    expect(decision.verdict).toBe("BLOCK");
    expect(decision.spans?.map((s) => text.slice(s.start, s.end))).toEqual(["Marcus Delgado"]);
  });

  it("does not allowlist a listed name introduced as someone's own contact", async () => {
    for (const text of ["My landlord John Smith won't fix the heat", "Mi casero John Smith no arregla la calefacción"]) {
      expect((await gate.check(text)).verdict).toBe("BLOCK");
    }
    expect(await gate.check("Write a story about Alexander Hamilton and my dog")).toMatchObject({ verdict: "ALLOW" });
  });

  it("still blocks an unlisted address and phone number", async () => {
    expect((await gate.check("Send it to 1601 Pennsylvania Avenue NW, Washington, DC 20500")).verdict).toBe("BLOCK");
    expect((await gate.check("Call me at 312-206-0199")).verdict).toBe("BLOCK");
  });

  it("adds host entries to the built-in lists", async () => {
    const text = "Write a story about Priya Ramanathan";
    expect((await gate.check(text)).verdict).toBe("BLOCK");
    const host = await loadLocalGate(undefined, MODEL, { names: ["Priya Ramanathan"] });
    expect(await host.check(text)).toMatchObject({ verdict: "ALLOW", basis: "allowlist" });
    expect(await host.check("Write a story about Alexander Hamilton")).toMatchObject({ verdict: "ALLOW" });
  });

  it("demotes a listed phone number in the validators", () => {
    const text = "Is 312-206-0199 the right number?";
    expect(createValidatorGate().validators(text).block).toBe(true);
    const result = createValidatorGate(undefined, { phones: ["(312) 206-0199"] }).validators(text);
    expect(result.block).toBe(false);
    expect(result.soft.find((d) => d.label === "PHONE")?.evidence).toMatch(/allowlist=host$/);
  });

  it("matches a listed address with or without the city, state and ZIP", () => {
    const list = compileAllowlist({ builtIn: false, addresses: ["300 Spring Garden St, Philadelphia, PA 19123"] });
    const clear = { block: false, hard: [], soft: [] };
    const covered = (text: string) => list.occurrences(text, clear).map((o) => text.slice(o.start, o.end));
    expect(covered("mail it to 300 Spring Garden Street")).toEqual(["300 Spring Garden Street"]);
    expect(covered("mail it to 300 Spring Garden St., Philadelphia PA 19123 today")).toEqual(["300 Spring Garden St., Philadelphia PA 19123"]);
    expect(covered("mail it to 301 Spring Garden St")).toEqual([]);
  });

  it("rejects entries that can never match", () => {
    expect(() => compileAllowlist({ phones: ["772-1213"] })).toThrow(/allowlist phone must be a complete number/);
    expect(() => compileAllowlist({ addresses: ["Spring Garden St"] })).toThrow(/allowlist address needs a number and a street/);
    expect(() => compileAllowlist({ names: ["  "] })).toThrow(/allowlist name has no letters/);
    expect(() => compileAllowlist({ names: "Alexander Hamilton" as never })).toThrow(/allowlist names must be an array of strings/);
  });
});
