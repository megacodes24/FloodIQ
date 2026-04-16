const state = {
  scenarios: [],
  defaultScenario: null,
  currentStudyArea: "lower_manhattan",
  forecastBoard: [],
  baselineByArea: {},
  map: null,
  segmentLayer: null,
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function riskColor(score) {
  if (score >= 0.75) return "#d7263d";
  if (score >= 0.6) return "#f46036";
  if (score >= 0.45) return "#ffb703";
  return "#ffd166";
}

function ensureMap(center) {
  if (state.map) return;
  state.map = L.map("map", {
    zoomControl: true,
    attributionControl: true,
  }).setView([center.lat, center.lon], 14);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);

  state.segmentLayer = L.layerGroup().addTo(state.map);
}

function inchesToMillimeters(value) {
  return value * 25.4;
}

function millimetersToInches(value) {
  return value / 25.4;
}

function renderBaseline(payload) {
  state.scenarios = payload.scenarios || [];
  state.defaultScenario = payload.default_scenario || payload.scenarios?.[0] || null;
  state.currentStudyArea = payload.study_area?.slug || "lower_manhattan";

  document.getElementById("studyAreaLabel").textContent = payload.study_area?.name || "Flood-prone NYC area";
  document.getElementById("forecastModeLabel").textContent = payload.weather?.forecast_available
    ? "Automatic from NOAA"
    : "Manual fallback";
  document.getElementById("scenarioModeLabel").textContent = payload.weather?.forecast_available
    ? "Forecast-linked"
    : "Manual storm";

  if (state.defaultScenario) {
    const forecastModeLabel = document.getElementById("forecastModeLabel");
    if (forecastModeLabel && state.defaultScenario.name === "Live NOAA forecast") {
      forecastModeLabel.textContent = "Automatic from NOAA";
    }
  }

  renderGn100Strip(payload);
  renderWeather(payload);
  highlightAreaCard();
}

function renderGn100Strip(payload) {
  const strip = document.getElementById("gn100Strip");
  strip.innerHTML = `
    <div class="stripItem">
      <span>Running on</span>
      <strong>Acer GN100 local runtime</strong>
    </div>
  `;
}

function renderWeather(payload) {
  const card = document.getElementById("weatherCard");
  const forecast = payload.weather?.active_forecast;
  if (forecast) {
    const millimetersPerHour = Math.round(inchesToMillimeters(forecast.rainfall_inches_per_hour));
    const leadIn = forecast.hours_until_start >= 24
      ? `Rainfall of ${millimetersPerHour} mm/hr is predicted on ${forecast.start_time_label}.`
      : `Rainfall of ${millimetersPerHour} mm/hr is predicted starting ${forecast.start_time_label}.`;
  card.innerHTML = `
      <div class="sectionTitle">
        <h2>Incoming Weather</h2>
        <p>Automatic flood trigger</p>
      </div>
      <p class="briefingLead">${leadIn}</p>
      <p class="briefingBody">FloodIQ is using the next available NOAA rain window directly and projecting likely flood risk onto streets and neighborhoods in the selected area. Forecast duration: <strong>${forecast.duration_hours} hr</strong>.</p>
      <p class="briefingMeta">${forecast.hours_until_start} hours until impact window | ${forecast.source_summary || "NOAA rain forecast"}</p>
    `;
    return;
  }

  card.innerHTML = `
    <div class="sectionTitle">
      <h2>Incoming Weather</h2>
      <p>Manual fallback</p>
    </div>
    <p class="briefingLead">No active rain forecast is available right now.</p>
    <p class="briefingBody">${payload.weather?.status || "NOAA forecast unavailable right now. Use manual rainfall controls to test likely street flooding."}</p>
  `;
}

function renderMap(mapPayload) {
  ensureMap(mapPayload.center);
  state.segmentLayer.clearLayers();

  const bounds = [
    [mapPayload.bounds.lat_min, mapPayload.bounds.lon_min],
    [mapPayload.bounds.lat_max, mapPayload.bounds.lon_max],
  ];
  state.map.fitBounds(bounds, { padding: [24, 24] });

  (mapPayload.risk_segments || []).forEach((segment) => {
    const color = riskColor(segment.risk_score);
    const line = L.polyline(segment.coordinates, {
      color,
      weight: segment.risk_score >= 0.65 ? 9 : 6,
      opacity: 0.9,
      lineCap: "round",
      className: segment.risk_score >= 0.65 ? "hotSegment hotSegmentStrong" : "hotSegment",
    });
    line.bindPopup(`
      <strong>${segment.name}</strong><br />
      Neighborhood: ${segment.neighborhood}<br />
      Risk score: ${segment.risk_score}<br />
      Peak depth: ${segment.max_depth_m} m
    `);
    line.addTo(state.segmentLayer);
  });
}

function renderMetrics(summary) {
  const metrics = document.getElementById("metrics");
  metrics.innerHTML = `
    <div class="metric"><span>Peak depth</span><strong>${summary.peak_depth_m} m</strong></div>
    <div class="metric"><span>Water volume</span><strong>${summary.water_volume_m3} m3</strong></div>
    <div class="metric"><span>Top street</span><strong>${summary.top_block}</strong></div>
    <div class="metric"><span>Street risk</span><strong>${summary.top_block_risk}</strong></div>
  `;
}

function renderOverlaySummary(summary, neighborhoods, forecast) {
  const forecastWindow = document.getElementById("overlayForecastWindow");
  const topNeighborhood = document.getElementById("overlayNeighborhood");
  const topStreet = document.getElementById("overlayStreet");

  forecastWindow.textContent = forecast?.start_time_label || "Forecast unavailable";
  topNeighborhood.textContent = neighborhoods?.[0]?.name || "Selected area";
  topStreet.textContent = summary.top_block || "Awaiting forecast";
  const actionBar = document.getElementById("mapActionBar");
  const topLocations = (neighborhoods || [])
    .slice(0, 2)
    .map((item) => item.locations?.[0])
    .filter(Boolean)
    .join(" | ");
  actionBar.innerHTML = `
    <div class="actionBarItem">
      <span>Watch now</span>
      <strong>${topLocations || summary.top_block || "Awaiting forecast"}</strong>
    </div>
    <div class="actionBarItem">
      <span>Peak water</span>
      <strong>${summary.peak_depth_m} m</strong>
    </div>
  `;
}

function renderPreRunState(payload) {
  const center = {
    lat: payload?.study_area?.slug === "lower_manhattan" ? 40.715 : 40.7128,
    lon: payload?.study_area?.slug === "lower_manhattan" ? -74.006 : -74.006,
  };
  ensureMap(center);
  state.segmentLayer.clearLayers();
  document.getElementById("overlayForecastWindow").textContent = payload?.weather?.active_forecast?.start_time_label || "Waiting for forecast run";
  document.getElementById("overlayNeighborhood").textContent = "No forecast run yet";
  document.getElementById("overlayStreet").textContent = "Press Run Latest Forecast";
  document.getElementById("mapActionBar").innerHTML = `
    <div class="actionBarItem">
      <span>Watch now</span>
      <strong>Awaiting forecast</strong>
    </div>
    <div class="actionBarItem">
      <span>Peak water</span>
      <strong>--</strong>
    </div>
  `;
  document.getElementById("actionsCard").innerHTML = `
    <div class="sectionTitle">
      <h2>Recommended Actions</h2>
      <p>What city teams can do next</p>
    </div>
    <p class="briefingLead">Run the latest forecast to generate dispatch actions.</p>
  `;
  document.getElementById("neighborhoodsCard").innerHTML = `
    <div class="sectionTitle">
      <h2>Neighborhood Watchlist</h2>
      <p>Forecasted flood concentration</p>
    </div>
    <p class="briefingLead">No neighborhoods are flagged until a forecast run is executed.</p>
  `;
  document.getElementById("briefingCard").innerHTML = `
    <div class="sectionTitle">
      <h2>Planner Briefing</h2>
      <p>Forecast-driven output</p>
    </div>
    <p class="briefingLead">The map is ready. Press <strong>Run Latest Forecast</strong> to compute flood-prone streets and neighborhoods.</p>
  `;
  document.getElementById("explanationCard").innerHTML = `
    <div class="sectionTitle">
      <h2>Why This Area</h2>
      <p>Risk drivers behind the alert</p>
    </div>
    <p class="briefingLead">Top-risk explanations will appear after a forecast run.</p>
  `;
  document.getElementById("metrics").innerHTML = `
    <div class="metric"><span>Peak depth</span><strong>--</strong></div>
    <div class="metric"><span>Water volume</span><strong>--</strong></div>
    <div class="metric"><span>Top street</span><strong>--</strong></div>
    <div class="metric"><span>Street risk</span><strong>--</strong></div>
  `;
  document.getElementById("alerts").innerHTML = `<div class="alert">Run the latest forecast to rank flood-prone streets.</div>`;
  document.getElementById("blockTableBody").innerHTML = "";
}

function renderForecastBoard(payload) {
  state.forecastBoard = payload.areas || [];
  const board = document.getElementById("forecastBoard");
  board.innerHTML = state.forecastBoard.map((area) => {
    const forecast = area.forecast;
    const forecastLine = forecast
      ? `${Math.round(inchesToMillimeters(forecast.rainfall_inches_per_hour))} mm/hr on ${forecast.start_time_label}`
      : (area.status || "No rain forecast in the current window");
    return `
      <button class="areaCard${area.slug === state.currentStudyArea ? " areaCardActive" : ""}" data-area-slug="${area.slug}">
        <span class="areaCardEyebrow">${area.subtitle}</span>
        <strong>${area.name}</strong>
        <p>${forecastLine}</p>
      </button>
    `;
  }).join("");

  board.querySelectorAll("[data-area-slug]").forEach((button) => {
    button.addEventListener("click", async () => {
      const slug = button.getAttribute("data-area-slug");
      if (!slug) return;
      state.currentStudyArea = slug;
      const boardArea = state.forecastBoard.find((area) => area.slug === slug);
      const baseline = state.baselineByArea[slug] || {
        study_area: { name: boardArea?.name || slug, slug },
        weather: {
          forecast_available: Boolean(boardArea?.forecast),
          active_forecast: boardArea?.forecast || null,
          status: boardArea?.status || "No rain forecast currently available",
        },
      };
      renderBaseline(baseline);
      renderPreRunState(baseline);
      highlightAreaCard();
    });
  });
}

function highlightAreaCard() {
  document.querySelectorAll("[data-area-slug]").forEach((button) => {
    const active = button.getAttribute("data-area-slug") === state.currentStudyArea;
    button.classList.toggle("areaCardActive", active);
  });
}

function renderBriefing(summary, solver, neighborhoods, forecast) {
  const card = document.getElementById("briefingCard");
  const topNeighborhoods = (neighborhoods || [])
    .slice(0, 2)
    .map((item) => item.name)
    .join(" and ");

  const action =
    summary.peak_depth_m >= 0.9
      ? "Pre-stage crews and inspect low-lying corridors before the forecast window begins."
      : summary.peak_depth_m >= 0.5
        ? "Monitor the top-ranked streets and prepare traffic diversions."
        : "Localized ponding is likely; monitor drains and vulnerable intersections.";

  card.innerHTML = `
    <div class="sectionTitle">
      <h2>Planner Briefing</h2>
      <p>Forecast-driven output</p>
    </div>
    <p class="briefingLead">${topNeighborhoods || "The selected area"} is projected to peak at <strong>${summary.peak_depth_m} m</strong> during the incoming rain window.</p>
    <p class="briefingBody">Top street-level hotspot: <strong>${summary.top_block}</strong>. ${action}</p>
    <p class="briefingMeta">Source: ${forecast?.name || "Manual rainfall"} | Model path: ${solver.engine}</p>
  `;
}

function renderExplanation(explanation) {
  const card = document.getElementById("explanationCard");
  const drivers = (explanation?.drivers || [])
    .map((driver) => `<div class="driverItem">${driver}</div>`)
    .join("");

  card.innerHTML = `
    <div class="sectionTitle">
      <h2>Why This Area</h2>
      <p>Risk drivers behind the alert</p>
    </div>
    <p class="briefingLead">${explanation?.headline || "Awaiting top-risk explanation."}</p>
    <p class="briefingMeta">${explanation?.neighborhood || ""}</p>
    <div class="driverList">${drivers || "<div class=\"driverItem\">No risk drivers available yet.</div>"}</div>
  `;
}

function renderAlerts(alerts, blocks) {
  document.getElementById("alerts").innerHTML = alerts
    .map((alert) => `<div class="alert">${alert}</div>`)
    .join("");

  document.getElementById("blockTableBody").innerHTML = blocks
    .slice(0, 8)
    .map(
      (block) => `
        <tr>
          <td>${block.name}</td>
          <td>${block.max_depth_m} m</td>
          <td>${block.risk_score}</td>
        </tr>
      `,
    )
    .join("");
}

function renderActions(actions) {
  const card = document.getElementById("actionsCard");
  const items = (actions || [])
    .map((action) => `
      <div class="actionItem">
        <strong>${action.title}</strong>
        <p>${action.detail}</p>
        <span class="actionMeta">${action.neighborhood || ""}</span>
      </div>
    `)
    .join("");

  card.innerHTML = `
    <div class="sectionTitle">
      <h2>Recommended Actions</h2>
      <p>What city teams can do next</p>
    </div>
    ${items || "<p class=\"briefingLead\">No recommended actions are available yet.</p>"}
  `;
}

function renderNeighborhoods(neighborhoods) {
  const card = document.getElementById("neighborhoodsCard");
  const items = (neighborhoods || [])
    .map(
      (item) => `
        <div class="actionItem">
          <strong>${item.name}</strong>
          <p>Peak depth ${item.peak_depth_m} m | average risk ${item.risk_score}</p>
          <span class="actionMeta">Watch: ${item.locations.join(", ")}</span>
        </div>
      `,
    )
    .join("");

  card.innerHTML = `
    <div class="sectionTitle">
      <h2>Neighborhood Watchlist</h2>
      <p>Forecasted flood concentration</p>
    </div>
    ${items || "<p class=\"briefingLead\">No neighborhood hotspots are available yet.</p>"}
  `;
}

async function runSimulation(mode = "forecast") {
  const body = {
    name: mode === "forecast" ? "Forecast-driven flood run" : "Manual flood run",
    study_area: state.currentStudyArea,
    refresh: mode === "forecast",
  };

  const payload = await fetchJson("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const refreshedBaseline = await loadBaseline(state.currentStudyArea, false);
  renderBaseline(refreshedBaseline);
  await loadForecastBoard(false);

  renderMap(payload.map);
  renderMetrics(payload.summary);
  renderOverlaySummary(payload.summary, payload.neighborhoods, payload.forecast_context);
  renderAlerts(payload.alerts, payload.blocks);
  renderActions(payload.recommended_actions);
  renderNeighborhoods(payload.neighborhoods);
  renderBriefing(payload.summary, payload.solver, payload.neighborhoods, payload.forecast_context);
  renderExplanation(payload.explanation);
}

async function loadBaseline(studyArea = state.currentStudyArea, refresh = false) {
  const baseline = await fetchJson(`/api/baseline?study_area=${encodeURIComponent(studyArea)}${refresh ? "&refresh=1" : ""}`);
  state.baselineByArea[studyArea] = baseline;
  renderBaseline(baseline);
  return baseline;
}

async function loadForecastBoard(refresh = false) {
  const payload = await fetchJson(`/api/forecast_board${refresh ? "?refresh=1" : ""}`);
  renderForecastBoard(payload);
}

async function boot() {
  await loadForecastBoard();
  const baseline = await loadBaseline("lower_manhattan");
  renderPreRunState(baseline);

  document.getElementById("runForecastButton").addEventListener("click", async () => {
    await runSimulation("forecast");
  });
}

boot().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<pre>${error.message}</pre>`;
});
