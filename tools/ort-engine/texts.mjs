// Synthetic test texts only. Nothing here is real user data.
const first = ["Maria","John","Aisha","Wei","Carlos","Fatima","Dmitri","Priya","Liam","Sofia","Kwame","Hannah","Ahmed","Elena","Tomas","Yuki","Grace","Omar","Ingrid","Mateo"];
const last = ["Garcia","Smith","Khan","Chen","Rodriguez","Hassan","Volkov","Patel","O'Brien","Rossi","Mensah","Schmidt","Ali","Petrova","Novak","Tanaka","Okafor","Dubois","Larsen","Silva"];
const streets = ["Maple Ave","Oak Street","Elm Dr","Pine Rd","Cedar Lane","Birch Blvd","Washington St","Lincoln Ave","Main Street","Park Pl"];
const cities = ["Springfield, IL","Austin, TX","Denver, CO","Portland, OR","Tampa, FL","Albany, NY","Boise, ID","Reno, NV","Toledo, OH","Fresno, CA"];
const plain = [
  "How do I renew my passport?",
  "What are the income limits for SNAP benefits this year?",
  "Where can I find the form to apply for a REAL ID?",
  "When is the deadline to file federal taxes?",
  "How do I check the status of my VA disability claim?",
  "Can you explain how Medicare Part D works?",
  "What documents do I need to apply for Social Security retirement?",
  "Is the Grand Canyon National Park open in winter?",
  "How does the FAFSA process work for returning students?",
  "What is the current federal minimum wage?",
  "Tell me about the history of the Lincoln Memorial.",
  "How do I report a lost or stolen passport?",
  "¿Cómo solicito una tarjeta de Seguro Social de reemplazo?",
  "Comment puis-je renouveler mon permis de conduire ?",
  "What is the weather usually like in Yellowstone in June?",
  "Explain the difference between a Roth IRA and a traditional IRA.",
  "How do I contact the IRS about a refund that has not arrived?",
  "What is Form I-130 used for?",
  "Can I vote by mail in the next election?",
  "How many national parks are there in the United States?",
];
function pad(n, w) { return String(n).padStart(w, "0"); }
export function buildTexts(count = 200) {
  const out = [];
  let i = 0;
  const pick = (arr) => arr[i++ % arr.length];
  while (out.length < count) {
    const k = out.length % 10;
    const name = `${pick(first)} ${pick(last)}`;
    const ssn = `${pad(100 + (i * 7) % 800, 3)}-${pad(10 + (i * 3) % 89, 2)}-${pad(1000 + (i * 13) % 8999, 4)}`;
    const phone = `(${pad(201 + (i * 5) % 700, 3)}) 555-${pad((i * 17) % 10000, 4)}`;
    const email = `${pick(first).toLowerCase()}.${pick(last).toLowerCase().replaceAll("'", "")}${i}@example.com`;
    const addr = `${100 + (i * 37) % 9800} ${pick(streets)}, ${pick(cities)} ${pad(10000 + (i * 101) % 89999, 5)}`;
    switch (k) {
      case 0: out.push(`my ssn is ${ssn}`); break;
      case 1: out.push(`You can reach me at ${phone} after 5pm.`); break;
      case 2: out.push(`Please send the forms to ${email}.`); break;
      case 3: out.push(`I live at ${addr} and need to update my address.`); break;
      case 4: out.push(`My name is ${name} and I need help with my benefits.`); break;
      case 5: out.push(pick(plain)); break;
      case 6: out.push(`Hi, I'm ${name}, SSN ${ssn}, phone ${phone}. Can you check my claim?`); break;
      case 7: out.push(pick(plain)); break;
      case 8: out.push(`My daughter ${pick(first)} was born on 03/${pad(1 + i % 28, 2)}/2015 and we moved to ${pick(cities)}.`); break;
      case 9: out.push(`Call ${phone} or email ${email}; my case number is A${pad(10000000 + i * 977, 9)}.`); break;
    }
  }
  return out;
}
