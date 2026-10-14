import { describe, expect, it } from "vitest";

import { detect, gate, type Label } from "../src/validators/index";

const hardLabels = (t: string) =>
  detect(t)
    .filter((d) => d.hard)
    .map((d) => d.label);

describe("Rampart Edge validators (r2 port)", () => {
  const positives: [string, Label][] = [
    ["My SSN is 452-78-1934", "SSN"],
    ["ITIN 912-70-1234", "ITIN"],
    ["Medicare 3CX7-MK2-PW45", "MEDICARE_MBI"],
    ["A-Number A087654321", "USCIS_A_NUMBER"],
    ["receipt IOE0987123456", "USCIS_RECEIPT"],
    ["routing number 071000013", "ROUTING_NUMBER"],
    ["4532 0151 1283 0366", "CREDIT_CARD"],
    ["DE89 1001 0010 0532 0130 07", "IBAN"],
    ["call me at (202) 456-0147", "PHONE"],
    ["+44 20 7946 0958", "PHONE_E164"],
    ["jane.doe@gmail.com", "EMAIL"],
    ["73.162.44.10", "IPV4"],
    ["passport number C48213957", "PASSPORT"],
    ["Is 123-45-6789 a valid SSN format?", "SSN"],
    ["SSN 452\u00a078\u00a01934", "SSN"],
    ["Is 1-800-772-1213 the Social Security number to call?", "PHONE"],
  ];
  expect(positives.length).toBeGreaterThan(10);
  for (const [text, label] of positives) {
    it(`hard-blocks ${label}: ${text}`, () => {
      expect(hardLabels(text)).toContain(label);
      expect(gate(text).block).toBe(true);
    });
  }

  const benign = [
    "Call the SSA hotline or dial 311 or 911",
    "Form I-485, HUD-52675, W-2, SF-86",
    "42 U.S.C. § 1983",
    "$9870 and $9,870.00 and 22%",
    "ZIP 20500 and 60680-4120",
    "Case No. 1:23-cv-04567",
    "Odometer 123,456 miles",
    "Tracking 9400 1000 0000 0000 0000 00",
    "The deadline is 04/15/2027",
    "appointment 2026-10-15 at 9:30",
    "Is the IRS on Pennsylvania Avenue open?",
    "version 10.0.19045.1",
    "127.0.0.1 and 192.168.1.1",
    "info@irs.gov",
    "foia@irs.gov",
    "https://www.ssa.gov/myaccount/",
    "OMB No. 1545-0074",
    "Red Cross EIN 53-0196605",
    "Does a passport number look like C12345678?",
    "sample shows 312-555-0199",
    "The passport renewal fee is $130 and processing takes 6 to 8 weeks.",
    "Total 2016\t416\t361\t1108\t2053",
    "e.g. IOE0912345678",
    "DOB 05/12/1980",
    "VIN 1M8GDM9AXKP042788",
  ];
  expect(benign.length).toBeGreaterThan(10);
  for (const text of benign) {
    it(`does not hard-block number-shaped benign text: ${JSON.stringify(text)}`, () => {
      expect(gate(text).block).toBe(false);
    });
  }

  it("keeps soft detections as evidence without blocking", () => {
    const result = gate("ZIP 20500");
    expect(result.block).toBe(false);
    expect(result.soft.map((d) => d.label)).toEqual(["ZIP"]);
  });

  it("demotes an allowlisted role mailbox and format example to soft with the list recorded", () => {
    const mailbox = detect("write to foia@irs.gov");
    expect(mailbox).toHaveLength(1);
    expect(mailbox[0].hard).toBe(false);
    expect(mailbox[0].evidence).toContain("role-mailbox");
    const receipt = detect("e.g. IOE0912345678");
    expect(receipt).toHaveLength(1);
    expect(receipt[0].hard).toBe(false);
    expect(receipt[0].evidence).toContain("allowlist=E1");
  });

  it("reads SSNs grouped by spaced slashes or double spaces", () => {
    expect(detect("SSN = 858 / 33 / 6165").map((d) => [d.label, d.value, d.hard])).toEqual([["SSN", "858336165", true]]);
    expect(detect("S S N: 227  55  6600").map((d) => [d.label, d.hard])).toEqual([["SSN", true]]);
  });

  it("treats any .gov or .mil address as soft", () => {
    expect(detect("email me at jsmith@irs.gov or j.doe@army.mil").map((d) => [d.label, d.hard])).toEqual([
      ["EMAIL", false],
      ["EMAIL", false],
    ]);
    expect(hardLabels("write to jsmith@irs.gov.example.com")).toEqual(["EMAIL"]);
  });

  it("treats an ombudsman mailbox as a role mailbox (soft)", () => {
    const [d] = detect("Should I email ombudsman@uscis.dhs.gov about my case delay?");
    expect(d.hard).toBe(false);
    expect(d.evidence).toContain("role-mailbox");
  });

  it("resolves overlaps so one span has one label", () => {
    expect(detect("IP 73.162.44.10").map((d) => d.label)).toEqual(["IPV4"]);
    expect(detect("4111 1111 1111 1111").map((d) => d.label)).toEqual(["CREDIT_CARD"]);
    expect(detect("123 45 6789 202 456 0147").map((d) => d.label)).toEqual(["SSN", "PHONE"]);
  });

  it("maps offsets back to the original string", () => {
    const text = "x 123\u200b45\u200b6789 y";
    const [ssn] = detect(text);
    expect(text.slice(ssn.start, ssn.end)).toBe("123\u200b45\u200b6789");
  });
});
