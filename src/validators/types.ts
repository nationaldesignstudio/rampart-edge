export type Label =
  | "SSN"
  | "ITIN"
  | "EIN"
  | "MEDICARE_MBI"
  | "USCIS_A_NUMBER"
  | "USCIS_RECEIPT"
  | "PASSPORT"
  | "MRZ"
  | "DRIVERS_LICENSE"
  | "ROUTING_NUMBER"
  | "BANK_ACCOUNT"
  | "CREDIT_CARD"
  | "IBAN"
  | "BIC"
  | "PHONE"
  | "PHONE_E164"
  | "EMAIL"
  | "URL"
  | "IPV4"
  | "IPV6"
  | "MAC"
  | "ZIP"
  | "DATE"
  | "DOB"
  | "VIN"
  | "LICENSE_PLATE"
  | "COORDINATES"
  | "ES_DNI"
  | "FR_NIR"
  | "DE_STEUER_ID"
  | "IT_CODICE_FISCALE"
  | "PT_NIF"
  | "NL_BSN"
  | "BR_CPF";

export interface Detection {
  label: Label;
  /** offsets into the ORIGINAL (un-normalized) string */
  start: number;
  end: number;
  /** canonical value (separators removed, upper-cased where alphabetic) */
  value: string;
  /** true = strong enough to block alone ("hard evidence"); false = soft feature for the model */
  hard: boolean;
  /** semicolon-separated machine-readable reasons, e.g. "luhn;iin=visa;len=16;fmt=4-4-4-4" */
  evidence: string;
  /** rule id that produced it */
  rule: string;
}

export interface Norm {
  /** normalized text: zero-width chars removed, NBSP/NNBSP/thin spaces -> " ", fullwidth digits -> ASCII */
  text: string;
  /** map[i] = index into original string for normalized char i */
  map: number[];
  original: string;
}

export interface RuleFn {
  (n: Norm): Detection[];
}
