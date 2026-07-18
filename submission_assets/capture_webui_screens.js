async (page) => {
  const out = "submission_assets/webui_video_capture";

  async function shot(name, locator = null) {
    await page.waitForTimeout(450);
    await (locator || page).screenshot({ path: `${out}/${name}.png` });
  }

  async function waitForBodyText(text, timeout = 45000) {
    await page.waitForFunction(
      (needle) => document.body.innerText.includes(needle),
      text,
      { timeout },
    );
  }

  await page.goto("http://127.0.0.1:8765/", { waitUntil: "networkidle" });
  await page.setViewportSize({ width: 1440, height: 1100 });
  await waitForBodyText("索引来源", 20000);
  await waitForBodyText("已索引", 20000).catch(() => {});

  const main = page.locator("main.wrap");
  const askPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "提问" }) });
  const answerPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "答案" }) });
  const sourcePanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "索引来源" }) });
  const statusPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "状态" }) });
  const actionPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "操作" }) });
  const reviewPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "待复核资料" }) });
  const apiPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "API 精炼" }) });
  const logPanel = page.locator(".panel").filter({ has: page.getByRole("heading", { name: "运行日志" }) });

  await shot("01_full_dashboard");
  await shot("02_status_panel", statusPanel);
  await shot("03_source_table", sourcePanel);
  await shot("04_operations_panel", actionPanel);
  await shot("05_review_panel", reviewPanel);
  await shot("06_api_panel", apiPanel);

  await page.fill("#question", "LogMsgType有哪些值");
  await shot("07_question_ready", askPanel);
  await page.click("#askBtn");
  await page.waitForFunction(
    () => {
      const meta = document.querySelector("#askMeta")?.textContent || "";
      const answer = document.querySelector("#answer")?.textContent || "";
      const sources = document.querySelector("#sources")?.textContent || "";
      return meta.length > 0 && answer.length > 30 && sources.length > 30 && !answer.includes("等待查询");
    },
    null,
    { timeout: 90000 },
  ).catch(() => {});
  await shot("08_answer_with_sources", main);
  await shot("09_answer_panel", answerPanel);
  await shot("10_log_panel", logPanel);

  await page.getByRole("button", { name: "清空" }).click();
  await page.fill("#question", "ZZZ999有哪些测试项？");
  await page.click("#askBtn");
  await page.waitForFunction(
    () => {
      const guidance = document.querySelector("#guidance")?.textContent || "";
      const answer = document.querySelector("#answer")?.textContent || "";
      return guidance.length > 0 || answer.includes("没有找到") || answer.includes("没有足够");
    },
    null,
    { timeout: 90000 },
  ).catch(() => {});
  await shot("11_no_evidence_guidance", main);

  if (!(await page.locator("#allowWeb").isChecked())) {
    await page.click("#allowWeb");
  }
  await shot("12_online_option", askPanel);
  await shot("13_final_full");
}
