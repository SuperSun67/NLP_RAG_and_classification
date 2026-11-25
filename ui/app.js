"use strict";

const $ = (id) => document.getElementById(id);
const form = $("review-form");
const input = $("review");
const examples = {
  positive: "อาหารอร่อยมาก วัตถุดิบสด บรรยากาศดี พนักงานดูแลเอาใจใส่ ราคาเหมาะสม มีโอกาสจะกลับมาอีกแน่นอน ประทับใจทุกจานเลยค่ะ",
  mixed: "อาหารรสชาติใช้ได้ บรรยากาศร้านโอเค แต่รอนานไปหน่อย พนักงานดูแลไม่ค่อยทั่วถึง ราคาค่อนข้างสูงเมื่อเทียบกับปริมาณ",
  negative: "ผิดหวังมาก อาหารไม่สดและเย็นชืด รอนานเกือบชั่วโมง พนักงานพูดจาไม่ดี ราคาแพงเกินคุณภาพ คงไม่กลับมาอีก",
};
let ready = false;
let busy = false;

function updateInput() {
  $("char-count").textContent = `${input.value.length.toLocaleString("en-US")} / 20,000`;
  $("submit").disabled = busy || !ready || !input.value.trim();
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  $("clear").disabled = value;
  document.querySelectorAll(".example").forEach((button) => { button.disabled = value; });
  document.querySelector(".result").setAttribute("aria-busy", String(value));
  $("loading-state").hidden = !value;
  $("submit-text").textContent = value ? "กำลังวิเคราะห์…" : "วิเคราะห์รีวิว";
  updateInput();
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function resetResult() {
  $("error").hidden = true;
  $("result-content").hidden = true;
  $("empty-state").hidden = false;
  $("result-announcement").textContent = "";
}

input.addEventListener("input", () => { updateInput(); resetResult(); });
$("clear").addEventListener("click", () => { input.value = ""; updateInput(); resetResult(); input.focus(); });
document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = examples[button.dataset.example];
    resetResult(); updateInput(); input.focus();
  });
});
input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("submit").disabled) form.requestSubmit();
  }
});

function renderResult(data) {
  $("score").textContent = data.stars;
  $("score-stars").replaceChildren();
  for (let i = 1; i <= 5; i++) {
    const star = document.createElement("span");
    star.textContent = "★";
    if (i > data.stars) star.className = "unfilled";
    $("score-stars").append(star);
  }
  $("score-caption").textContent = `โมเดลทำนาย ${data.stars} ดาว · ความน่าจะเป็น ${(data.probabilities[String(data.stars)] * 100).toFixed(1)}%`;
  $("time").textContent = data.elapsed_ms < 1000 ? `${Math.round(data.elapsed_ms)} ms` : `${(data.elapsed_ms / 1000).toFixed(2)} s`;
  $("distribution").replaceChildren();
  for (let rating = 1; rating <= 5; rating++) {
    const percent = data.probabilities[String(rating)] * 100;
    const row = document.createElement("div");
    row.className = `probability-row${rating === data.stars ? " winner" : ""}`;
    const label = document.createElement("span");
    label.textContent = `${rating} ดาว`;
    const track = document.createElement("div");
    track.className = "bar-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = `${percent.toFixed(1)}%`;
    row.append(label, track, value);
    $("distribution").append(row);
    requestAnimationFrame(() => { fill.style.width = `${percent}%`; });
  }
  $("truncation").hidden = !data.truncated;
  $("truncation").textContent = `รีวิวนี้ยาว ${data.token_count.toLocaleString("en-US")} tokens โมเดลใช้เฉพาะช่วงต้นที่ตัดให้เหลือ ${data.used_tokens} tokens ในการวิเคราะห์`;
  $("empty-state").hidden = true;
  $("result-content").hidden = false;
  $("result-announcement").textContent = `วิเคราะห์เสร็จแล้ว โมเดลทำนาย ${data.stars} จาก 5 ดาว`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !ready || !input.value.trim()) return;
  $("error").hidden = true;
  $("empty-state").hidden = true;
  $("result-content").hidden = true;
  setBusy(true);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: input.value.trim() }), signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      const message = typeof data.detail === "string" ? data.detail : "กรุณาตรวจข้อความรีวิว (ไม่เกิน 20,000 ตัวอักษร) แล้วลองใหม่";
      throw new Error(message);
    }
    renderResult(data);
  } catch (error) {
    $("empty-state").hidden = false;
    showError(error.name === "AbortError" ? "ใช้เวลาประมวลผลนานกว่าปกติ กรุณารอสักครู่แล้วลองใหม่" : (error instanceof TypeError ? "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ กรุณาตรวจว่าแอปยังทำงานอยู่" : error.message));
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
});

async function checkStatus() {
  try {
    const response = await fetch("/api/status", { signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error("status unavailable");
    const data = await response.json();
    ready = data.ready;
    $("status").className = `status ${ready ? "ready" : "offline"}`;
    $("status-text").textContent = ready ? "โมเดลพร้อมใช้งาน" : "โมเดลยังไม่พร้อม";
    if (ready) $("error").hidden = true;
    if (!ready) showError(data.error || "ยังไม่มีโมเดลสำหรับวิเคราะห์");
    $("accuracy").textContent = data.evaluation ? `${(data.evaluation.accuracy * 100).toFixed(2)}%` : "—";
    $("f1").textContent = data.evaluation ? `${(data.evaluation.macro_f1 * 100).toFixed(2)}%` : "—";
  } catch {
    ready = false;
    $("status").className = "status offline";
    $("status-text").textContent = "ยังเชื่อมต่อไม่ได้";
    showError("เชื่อมต่อเซิร์ฟเวอร์ไม่ได้ กรุณาตรวจว่าแอปยังทำงานอยู่ ระบบจะลองเชื่อมต่ออีกครั้ง");
    setTimeout(checkStatus, 10000);
  }
  updateInput();
}

updateInput();
checkStatus();
