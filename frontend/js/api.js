/**
 * Smart Inventory Advisor — shared frontend API client.
 *
 * All requests go through `api()`, which automatically attaches the JWT
 * (if present) and throws a normalized Error on non-2xx responses so page
 * scripts can just try/catch.
 */

// If the frontend is served by the FastAPI app itself (StaticFiles mount),
// same-origin "/api" works. If you serve the frontend separately (e.g. a
// static host), change this to the full backend URL, e.g.
// "https://api.yourdomain.com/api".
const API_BASE = "/api";

function getToken() {
  return localStorage.getItem("sia_token");
}

function getStoreId() {
  return localStorage.getItem("sia_store_id");
}

function getUsername() {
  return localStorage.getItem("sia_username");
}

function setSession({ access_token, store_id, username }) {
  localStorage.setItem("sia_token", access_token);
  localStorage.setItem("sia_store_id", store_id);
  localStorage.setItem("sia_username", username);
}

function clearSession() {
  localStorage.removeItem("sia_token");
  localStorage.removeItem("sia_store_id");
  localStorage.removeItem("sia_username");
}

function isLoggedIn() {
  return !!getToken();
}

/** Redirect to login if there's no session. Call at the top of protected pages. */
function requireAuth() {
  if (!isLoggedIn()) {
    window.location.href = "login.html";
  }
}

/**
 * Core request helper.
 * @param {string} path - path relative to API_BASE, e.g. "/login"
 * @param {object} opts - { method, body, isForm }
 */
async function api(path, opts = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let body = opts.body;
  if (body && !opts.isForm) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: opts.method || "GET",
      headers,
      body,
    });
  } catch (networkErr) {
    throw new Error(
      "Could not reach the server. Check that the backend is running and try again."
    );
  }

  if (res.status === 401) {
    clearSession();
    if (!location.pathname.endsWith("login.html")) {
      window.location.href = "login.html";
    }
    throw new Error("Session expired. Please log in again.");
  }

  let data = null;
  const text = await res.text();
  if (text) {
    try { data = JSON.parse(text); } catch { data = text; }
  }

  if (!res.ok) {
    const detail = (data && data.detail) ? data.detail : `Request failed (${res.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  return data;
}

/** Populates the shared top navigation bar on every page. */
function renderNav(activePage) {
  const mount = document.getElementById("nav-mount");
  if (!mount) return;

  const loggedIn = isLoggedIn();
  const links = [
    { href: "index.html", label: "Home", key: "home" },
    { href: "upload.html", label: "Upload", key: "upload", auth: true },
    { href: "predict.html", label: "Predict", key: "predict", auth: true },
    { href: "dashboard.html", label: "Dashboard", key: "dashboard", auth: true },
  ];

  const navLinks = links
    .filter((l) => !l.auth || loggedIn)
    .map(
      (l) =>
        `<a href="${l.href}" class="${l.key === activePage ? "active" : ""}">${l.label}</a>`
    )
    .join("");

  const rightSide = loggedIn
    ? `<span class="store-pill mono">STORE ${getStoreId()}</span>
       <button class="btn-logout" id="logout-btn">Log out</button>`
    : `<a href="login.html" class="btn btn-amber" style="padding:8px 16px;">Log in</a>`;

  mount.innerHTML = `
    <div class="topbar">
      <div class="topbar-inner">
        <a href="index.html" class="brand"><span class="tag-icon"></span>Smart Inventory Advisor</a>
        <nav class="mainnav">${navLinks}</nav>
        <div style="display:flex;align-items:center;gap:10px;">${rightSide}</div>
      </div>
    </div>`;

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      clearSession();
      window.location.href = "index.html";
    });
  }
}

function showAlert(mountId, message, type = "error") {
  const el = document.getElementById(mountId);
  if (!el) return;
  el.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
}

function clearAlert(mountId) {
  const el = document.getElementById(mountId);
  if (el) el.innerHTML = "";
}

function fmtNum(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}
