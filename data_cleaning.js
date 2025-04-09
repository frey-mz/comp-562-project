// This file should not exist Matthew!

import fs from "node:fs";

let s = fs.readFileSync("output.jsonl");

let set = new Set();

let loads = "";

for (const line of s.toString().split("\n")) {
  const x = JSON.parse(line);
  if (set.has(x.id)) {
    console.log("dupe found of " + x.id);
    continue;
  }
  set.add(x.id);
  loads += line + "\n";
}

fs.writeFileSync("output_cleaned.jsonl", loads, "utf-8");

console.log("Done!");
