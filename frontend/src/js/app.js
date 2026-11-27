// Rag Search Lab: connects the page to the backend API.
//
// Flow of one search:
//   click Search -> read the form -> POST /api/search -> draw the result cards
//                -> POST /api/answer -> draw the answer and its citation chips
//
// Rules this file follows:
//   * Text from the server is only ever put on the page with .textContent, never innerHTML,
//     so a document containing "<script>" can never run code in the page.
//   * The Search button stays disabled while a request is running, so a double click cannot
//     start two searches (every answer costs money).
//   * Every request has a timeout, and every failure ends in a visible message.

// "" means "the same address that served this page". If the page is served from somewhere else,
// put the backend address here, e.g. "http://127.0.0.1:8000", and start the server with
// ALLOWED_ORIGINS set to the page's address.
const API_BASE = "";
const SEARCH_TIMEOUT_MS = 15_000;
const ANSWER_TIMEOUT_MS = 60_000;
const HIGHLIGHT_MS = 1_800;

const $ = (selector) => document.querySelector(selector);

const els = {
    input: $("#question"),
    searchButton: $(".search-button"),
    methodButtons: [...document.querySelectorAll(".method-button")],
    resultsCount: $("#results-count"),
    error: $(".error-message"),
    errorText: $(".error-message__text"),
    resultsSection: $(".results"),
    resultsEmpty: $(".results .empty-state"),
    skeleton: $(".results .skeleton"),
    answerEmpty: $(".answer .empty-state"),
    resultTemplate: $("#result-template"),
    answerTemplate: $("#answer-template"),
    noResultsTemplate: $("#no-results-template"),
};

// Two containers that this script fills and clears. They are created here, not in the HTML.
const resultsList = document.createElement("div");
resultsList.className = "results-list";
$(".results").append(resultsList);

const answerBody = document.createElement("div");
$(".answer").append(answerBody);

const cards = new Map(); // chunk_id -> result card element, so a citation chip can find its card
let busy = false;

// ---------------------------------------------------------------- talking to the server
class ApiError extends Error {
    constructor(status) {
        super(`HTTP ${status}`);
        this.status = status;
    }
}

async function postJson(path, body, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(API_BASE + path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        if (!response.ok) throw new ApiError(response.status);
        return await response.json();
    } finally {
        clearTimeout(timer);
    }
}

/** Turn any failure of the search request into a sentence for the red banner. */
function searchErrorText(error) {
    if (error instanceof ApiError && error.status === 422) {
        return "Please check your question. It must be between 1 and 500 characters.";
    }
    if (error instanceof ApiError) return "The server had a problem. Please try again.";
    if (error.name === "AbortError") return "The server took too long to answer. Please try again.";
    return "Could not reach the server. Check that the backend is running and try again.";
}

/** Same idea for the answer request. The result cards stay on screen when only this fails. */
function answerErrorText(error) {
    if (error instanceof ApiError && error.status === 503) {
        return "Answers are not set up on the server yet (no API key). The passages on the left are still valid.";
    }
    if (error instanceof ApiError && error.status === 502) {
        return "The answer service failed. The passages on the left are still valid. Please try again.";
    }
    if (error.name === "AbortError") return "The answer took too long. The passages on the left are still valid.";
    return "Could not get an answer. The passages on the left are still valid.";
}

// ---------------------------------------------------------------- drawing the page
function showError(text) {
    els.errorText.textContent = text;
    els.error.hidden = false;
}

function clearResultsAndAnswer() {
    resultsList.replaceChildren();
    answerBody.replaceChildren();
    cards.clear();
    els.error.hidden = true;
}

function setLoading(isLoading) {
    busy = isLoading;
    els.searchButton.disabled = isLoading;
    els.skeleton.hidden = !isLoading;
    els.resultsSection.setAttribute("aria-busy", String(isLoading));
    if (isLoading) els.resultsEmpty.hidden = true;
}

function buildResultCard(result) {
    const card = els.resultTemplate.content.firstElementChild.cloneNode(true);
    card.tabIndex = -1; // lets a citation chip move focus here
    card.querySelector(".rank-badge").textContent = `#${result.rank}`;
    card.querySelector(".result-card__title").textContent = result.title;
    card.querySelector(".result-card__score").textContent = result.score.toFixed(3);
    card.querySelector(".score-bar__fill").style.width = `${Math.round(result.relative_score * 100)}%`;
    card.querySelector(".result-card__snippet").textContent = result.text;
    card.querySelector(".result-card__source").textContent = result.source;

    const page = card.querySelector(".result-card__page");
    if (result.page == null) page.remove(); // the server does not track pages per passage yet
    else page.textContent = `Page ${result.page}`;

    // "Show original quote" needs a backend endpoint that does not exist yet, so no dead button.
    card.querySelector(".link-button").remove();
    return card;
}

function renderResults(results) {
    resultsList.replaceChildren();
    cards.clear();
    if (results.length === 0) {
        resultsList.append(els.noResultsTemplate.content.cloneNode(true));
        return;
    }
    for (const result of results) {
        const card = buildResultCard(result);
        cards.set(result.chunk_id, card);
        resultsList.append(card);
    }
}

function highlightCard(chunkId) {
    const card = cards.get(chunkId);
    if (!card) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    card.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    card.focus({ preventScroll: true });
    card.classList.add("result-card--highlight");
    setTimeout(() => card.classList.remove("result-card--highlight"), HIGHLIGHT_MS);
}

function showAnswerMessage(text) {
    els.answerEmpty.hidden = true;
    const message = document.createElement("p");
    message.className = "answer__text";
    message.textContent = text;
    answerBody.replaceChildren(message);
}

function renderAnswer(answer) {
    els.answerEmpty.hidden = true;
    const fragment = els.answerTemplate.content.cloneNode(true);
    fragment.querySelector(".answer__text").textContent = answer.truncated
        ? `${answer.text} (The answer was cut off.)`
        : answer.text;

    const sources = fragment.querySelector(".answer__sources");
    const chipTemplate = sources.querySelector(".citation-chip");
    chipTemplate.remove();
    for (const citation of answer.citations) {
        const chip = chipTemplate.cloneNode(true);
        chip.textContent = `[${citation.number}]`;
        chip.setAttribute("aria-label", `Show source ${citation.number}`);
        chip.addEventListener("click", () => highlightCard(citation.chunk_id));
        sources.append(chip);
    }
    if (answer.citations.length === 0) sources.remove(); // e.g. "I don't know": nothing to cite
    answerBody.replaceChildren(fragment);
}

// ---------------------------------------------------------------- one search, start to end
function activeMethod() {
    const pressed = els.methodButtons.find((b) => b.getAttribute("aria-pressed") === "true");
    return pressed.textContent.trim().toLowerCase(); // "Vector" | "BM25" | "Hybrid" -> vector | bm25 | hybrid
}

async function loadAnswer(request) {
    showAnswerMessage("Writing the answer…");
    answerBody.firstElementChild.setAttribute("role", "status");
    try {
        const data = await postJson("/api/answer", request, ANSWER_TIMEOUT_MS);
        renderAnswer(data.answer);
    } catch (error) {
        showAnswerMessage(answerErrorText(error));
    }
}

async function runSearch() {
    if (busy) return;
    const question = els.input.value.trim();
    if (!question) {
        els.input.focus();
        return;
    }
    const request = { question, method: activeMethod(), k: Number(els.resultsCount.value) };

    clearResultsAndAnswer();
    els.answerEmpty.hidden = false;
    setLoading(true);
    try {
        const found = await postJson("/api/search", request, SEARCH_TIMEOUT_MS);
        els.skeleton.hidden = true;
        renderResults(found.results);
        await loadAnswer(request);
    } catch (error) {
        els.resultsEmpty.hidden = false; // nothing to show: back to the starting message
        showError(searchErrorText(error));
    } finally {
        setLoading(false);
    }
}

// ---------------------------------------------------------------- wiring
els.searchButton.addEventListener("click", runSearch);
els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.isComposing) runSearch();
});
for (const button of els.methodButtons) {
    button.addEventListener("click", () => {
        for (const other of els.methodButtons) other.setAttribute("aria-pressed", String(other === button));
    });
}

// Tell the user straight away if the backend is not running.
fetch(`${API_BASE}/api/health`).catch(() => showError(searchErrorText(new TypeError("offline"))));
