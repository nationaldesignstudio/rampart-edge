import type { Norm } from "./types.js";

/** Keyword context lists (EN/ES/FR/DE/IT/PT/NL). Lower-cased, matched in a window before (and shortly after) the span. */
export const KW: Record<string, RegExp> = {
  SSN: /\b(ssn|ss#|ss ?#|social security|social-security|seguro social|n[uú]mero de seguro|sécurité sociale|sozialversicherung|previdenza|segurança social|sofinummer|burgerservice)/i,
  ITIN: /\b(itin|individual taxpayer|taxpayer identification|n[uú]mero de identificaci[oó]n personal del contribuyente)/i,
  EIN: /\b(ein|fein|employer identification|tax id|tax-id|taxpayer id|n[uú]mero de identificaci[oó]n del empleador)/i,
  MBI: /\b(mbi|medicare|beneficiary identifier|medicare number|n[uú]mero de medicare)/i,
  ANUMBER:
    /\b(a-?number|a ?#|alien (registration )?number|uscis ?#|uscis number|n[uú]mero a|n[uú]mero de extranjero|registro de extranjero)/i,
  RECEIPT: /\b(receipt|case number|case no|n[uú]mero de recibo|n[uú]mero de caso|uscis)/i,
  PASSPORT: /\b(passport|pasaporte|passeport|reisepass|passaporto|passaporte|paspoort)/i,
  DL: /\b(driver'?s? licen[cs]e|drivers licen[cs]e|dl ?#|dl number|license (no|number|#)|licencia de (conducir|manejar)|permis de conduire|führerschein|fuehrerschein|patente|carta de condução|rijbewijs|id card number|state id)/i,
  ROUTING: /\b(routing|aba|rtn|transit number|n[uú]mero de ruta|tránsito)/i,
  ACCOUNT:
    /\b(account(?: (no|number|#))?|acct|bank account|checking account|savings account|n[uú]mero de cuenta|cuenta bancaria|numéro de compte|kontonummer|numero di conto|número de conta|rekeningnummer|iban)/i,
  CARD: /\b(card|tarjeta|carte|karte|carta|cartão|kaart|visa|mastercard|amex|american express|discover|cvv|cvc|expir)/i,
  BIC: /\b(bic|swift)/i,
  PHONE:
    /\b(phone|tel|cell|mobile|call me|text me|whatsapp|tel[eé]fono|celular|móvil|movil|téléphone|portable|telefon|handy|telefono|cellulare|telemóvel|telefoon|mobiel|bel me)/i,
  /** first-person / private-person phone cue: promotes a NANP number to hard regardless of other context */
  PHONE_PERSONAL:
    /\b(call me|text me|reach me|ring me|mine is|that'?s mine|it'?s mine|el m[ií]o es|la m[ií]a es|le mien|la mienne|meine ist|il mio [eè]|o meu [eé]|de mijne|my (?:\w+[ -]){0,2}(?:cell|phone|number|mobile|tel|landline|home number|work number)|my (?:husband|wife|son|daughter|mom|mother|dad|father|landlord|boss|friend|neighbor|neighbour)'?s? (?:cell|phone|number|mobile)|ll[aá]ma(?:me|rme)|mi (?:\w+ ){0,2}(?:celular|n[uú]mero|tel[eé]fono|m[oó]vil)|appelez-moi|mon (?:num[eé]ro|portable|t[eé]l[eé]phone)|ruf(?:en sie)? mich|mein(?:e)? (?:nummer|handy|telefon)|chiamami|il mio (?:numero|cellulare|telefono)|me liga|meu (?:n[uú]mero|celular|telefone)|bel me|mijn (?:nummer|telefoon|mobiel))/i,
  /** public-entity cue: agency/office/business framing around a number leaves it to the model */
  PUBLIC_ENTITY:
    /\b(agency|agencies|office|offices|department|dept\.?|bureau|administration|division|commission|hotline|helpline|help line|call center|customer service|switchboard|main line|toll[- ]free|for (?:more )?information|more information|contact (?:us|them|the|your)|visit|website|www\.|\.gov|\.mil|\.org|hospital|clinic|library|court|county|city of|state of|university|school district|embassy|consulate|is (?:this|that) the (?:right )?number|n[uú]mero (?:de|del|de la) (?:la )?(?:agencia|oficina|departamento|l[ií]nea|servicio)|oficina|agencia|departamento|l[ií]nea (?:gratuita|de ayuda)|atenci[oó]n al cliente|bureau des|minist[eè]re|service public|standard|amt|beh[oö]rde|ministerium|b[uü]rgertelefon|ufficio|agenzia|ministero|numero verde|ag[eê]ncia|reparti[cç][aã]o|minist[eé]rio|kantoor|gemeente|ministerie|loket)/i,
  DOB: /\b(dob|d\.o\.b|date of birth|birth ?date|birthday|born|fecha de nacimiento|nacimiento|naci[oó]|cumplea[ñn]os|date de naissance|né le|née le|naissance|geburtsdatum|geboren|geburtstag|data di nascita|nato il|nata il|data de nascimento|nasceu|geboortedatum|geboren op)/i,
  VIN: /\b(vin|vehicle identification|chassis|n[uú]mero de serie del veh|fahrgestellnummer|numero di telaio|chassi|chassisnummer)/i,
  PLATE:
    /\b(license plate|licence plate|plates?\b|plate (no|number|#)|tag number|placa|matr[ií]cula|plaque d'immatriculation|immatriculation|kennzeichen|targa|kenteken)/i,
  COORD:
    /\b(coordinates|coordenadas|coordonnées|koordinaten|coordinate|coördinaten|lat|lon|lng|gps|latitude|longitude)/i,
  DNI: /\b(dni|nie|nif|documento nacional)/i,
  NIR: /\b(nir|insee|sécurité sociale|securite sociale|numéro de sécu|n° de sécu|numero de secu)/i,
  STEUER: /\b(steuer-?id|steueridentifikationsnummer|idnr|identifikationsnummer|steuernummer)/i,
  CF: /\b(codice fiscale|cod\.? ?fisc)/i,
  NIF_PT: /\b(nif|contribuinte|n[uú]mero fiscal|número de identificação fiscal)/i,
  BSN: /\b(bsn|burgerservicenummer|sofinummer|sofi-nummer)/i,
  CPF: /\b(cpf|cadastro de pessoa)/i,
  IP: /\b(ip|ip address|ipv4|ipv6|dirección ip|adresse ip)/i,
  ZIP: /\b(zip|zip code|postal code|código postal|code postal|postleitzahl|plz|cap|postcode)/i,
};

export function hasKw(
  n: Norm,
  re: RegExp,
  start: number,
  end: number,
  before = 48,
  after = 16,
): boolean {
  const pre = n.text.slice(Math.max(0, start - before), start);
  const post = n.text.slice(end, Math.min(n.text.length, end + after));
  return re.test(pre) || re.test(post);
}

/** format-question cue: the writer is asking what an identifier looks like, not supplying one. Not a KW entry so the feature schema stays fv1. */
export const FORMAT_EXAMPLE =
  /\b(looks? like|format|formato|example|ejemplo|exemple|beispiel|esempio|exemplo|voorbeeld|sample|e\.g\.|such as|placeholder|dummy)\b/i;
