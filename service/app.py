"""FastAPI application: pages, live-run API and result files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from service import __version__
from service.auth import check_basic_auth
from service.catalog import RESULT_FILES, Catalog, Trial, movement_rank, read_json, safe_file
from service.config import Settings
from service.insights import analyse, chart_svg, summary_rows, tracking_error_deg
from service.metrics_view import (
    accuracy_rows,
    calibration_rows,
    headline,
    session,
    stage_rows,
    sync_rows,
)
from service.metrics_view import overview as overview_figures
from service.runs import PRESETS, RunManager
from service.telemetry import log_event, setup_logging

HERE = Path(__file__).resolve().parent

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; media-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
PUBLIC_PATHS = ("/healthz", "/static/")
DOWNLOADS = ("motion.bvh", "skeleton3d.npz", "metrics.json", "calibration.json")
LIVE_PRESET = "quick"
RUN_MINUTES = 6.5  # wall-clock minutes of one live run, as quoted to visitors

EMPTY_RESULT: dict[str, Any] = {
    "session": None,
    "accuracy": [],
    "calibration": [],
    "sync": [],
    "stages": [],
    "videos": [],
    "downloads": [],
    "knee_svg": None,
    "knee_rows": [],
    "knee_error": None,
    "character": None,
}


def published(catalog: Catalog) -> list[tuple[Trial, dict[str, Any]]]:
    """Trials with a precomputed result and their metrics, in the order the results page lists them."""
    found = []
    for trial in catalog.trials():
        if trial.showcase_dir is not None:
            found.append((trial, read_json(trial.showcase_dir / "metrics.json") or {}))
    return sorted(found, key=lambda item: (movement_rank(item[0].movement_key), item[0].name))


def movement_groups(items: list[tuple[Trial, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Results grouped by movement type in page order. Movements without results are left out."""
    groups: dict[str, dict[str, Any]] = {}
    for trial, metrics in items:
        group = groups.setdefault(
            trial.movement_key, {"key": trial.movement_key, "label": trial.movement, "cards": []}
        )
        meta = session(metrics)
        group["cards"].append(
            {
                "trial": trial,
                "cameras": meta["cameras"] if meta else str(len(trial.cams)),
                "headline": headline(metrics),
            }
        )
    return [groups[key] for key in sorted(groups, key=movement_rank)]


def first_with(items: list[tuple[Trial, dict[str, Any]]], filename: str) -> Trial | None:
    """The first published trial that has ``filename``, otherwise the first published trial."""
    for trial, _ in items:
        if trial.showcase_dir is not None and (trial.showcase_dir / filename).is_file():
            return trial
    return items[0][0] if items else None


def knee_context(skeleton: Path, mocap: Path | None) -> dict[str, Any]:
    """Movement analysis for a result page. Empty when there is no readable skeleton."""
    try:
        result = analyse(skeleton, mocap)
    except Exception as exc:  # a damaged result file must not take the whole page down
        log_event("knee analysis failed", skeleton=str(skeleton), error=str(exc))
        result = None
    if result is None:
        return {"knee_svg": None, "knee_rows": [], "knee_error": None}
    return {
        "knee_svg": Markup(chart_svg(result)),
        "knee_rows": summary_rows(result),
        "knee_error": tracking_error_deg(result),
    }


def result_context(base: Path, mocap: Path | None) -> dict[str, Any]:
    """Everything the result sections show, read from one result folder."""
    metrics = read_json(base / "metrics.json") or {}
    return {
        "session": session(metrics),
        "accuracy": accuracy_rows(metrics),
        "calibration": calibration_rows(metrics),
        "sync": sync_rows(metrics),
        "stages": stage_rows(metrics),
        "videos": [name for name in ("overlay_2d.mp4", "skeleton_vs_mocap.mp4") if (base / name).is_file()],
        "downloads": [name for name in DOWNLOADS if (base / name).is_file()],
        "character": "character.mp4" if (base / "character.mp4").is_file() else None,
        **knee_context(base / "skeleton3d.npz", mocap),
    }


def run_cost(hourly_usd: float) -> str | None:
    """Estimated cost of one live run at the on-demand instance price; None when no price is set."""
    if hourly_usd <= 0:
        return None
    return f"{RUN_MINUTES * hourly_usd / 60:.4f}"


def create_app(settings: Settings | None = None, manager: RunManager | None = None) -> FastAPI:
    """Build the application. ``manager`` can be injected for tests."""
    settings = settings or Settings.from_env()
    settings.validate()
    setup_logging(settings.log_dir)
    catalog = Catalog(settings.data_dir, settings.showcase_dir)
    manager = manager or RunManager(
        settings.runs_dir, settings.repo_root, settings.python, settings.run_timeout_s,
        s3_bucket=settings.s3_bucket, cloudwatch=settings.cloudwatch,
    )

    app = FastAPI(title="AIMoCap", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals.update(version=__version__)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Response:
        path = request.url.path
        if not path.startswith(PUBLIC_PATHS) and not check_basic_auth(
            request.headers.get("authorization"), settings.password
        ):
            response: Response = Response(
                "Authentication required", status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="AIMoCap demo", charset="UTF-8"'},
            )
        else:
            response = await call_next(request)
        for key, value in SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    def page(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/", response_class=HTMLResponse)
    def overview(request: Request) -> HTMLResponse:
        items = published(catalog)
        links = {
            "use": first_with(items, "motion.bvh"),
            "movement": first_with(items, "skeleton3d.npz"),
            "accuracy": first_with(items, "metrics.json"),
        }
        return page(request, "overview.html", current="overview",
                    stats=overview_figures([metrics for _, metrics in items]), links=links,
                    active=manager.active())

    @app.get("/results", response_class=HTMLResponse)
    def results(request: Request) -> HTMLResponse:
        return page(request, "results.html", current="results", groups=movement_groups(published(catalog)))

    @app.get("/results/{name}", response_class=HTMLResponse)
    def trial_result(request: Request, name: str) -> HTMLResponse:
        trial = catalog.get(name)
        if trial is None or trial.showcase_dir is None:
            raise HTTPException(404, "Unknown result")
        context = result_context(trial.showcase_dir, trial.mocap_file())
        return page(request, "trial.html", current="results", trial=trial, kind="showcase", key=name,
                    run=None, **context)

    @app.get("/showcase/{name}")
    def showcase_redirect(name: str) -> RedirectResponse:
        """Former address of a precomputed result, kept as a redirect."""
        trial = catalog.get(name)
        if trial is None or trial.showcase_dir is None:
            raise HTTPException(404, "Unknown result")
        return RedirectResponse(f"/results/{name}", status_code=307)

    @app.get("/live", response_class=HTMLResponse)
    def live(request: Request) -> HTMLResponse:
        runnable = [trial for trial in catalog.trials() if catalog.runnable(trial)]
        return page(request, "live.html", current="live", trials=runnable, preset=PRESETS[LIVE_PRESET],
                    active=manager.active(), cost=run_cost(settings.hourly_usd))

    @app.post("/runs")
    def start_run(request: Request, trial: str = Form(...), preset: str = Form(LIVE_PRESET)) -> Response:
        found = catalog.get(trial)
        if found is None or not catalog.runnable(found):
            raise HTTPException(404, "Unknown trial")
        if preset not in PRESETS:
            raise HTTPException(400, "Unknown preset")
        try:
            run = manager.start(found, PRESETS[preset])
        except RuntimeError as exc:
            return page(request, "message.html", status_code=409, current="live", title="A run is in progress",
                        message=str(exc), link=f"/runs/{manager.active().id}" if manager.active() else "/live")
        return RedirectResponse(f"/runs/{run.id}", status_code=303)

    @app.get("/runs", response_class=HTMLResponse)
    def run_history(request: Request) -> HTMLResponse:
        rows = []
        for run in manager.list():
            cost = None
            if settings.hourly_usd > 0 and run.seconds:
                cost = f"{run.seconds / 3600 * settings.hourly_usd:.4f}"
            rows.append({"run": run, "cost": cost})
        return page(request, "runs.html", current="live", rows=rows)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str) -> HTMLResponse:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run")
        trial = catalog.get(run.trial)
        context = dict(EMPTY_RESULT)
        if run.status == "completed":
            context = result_context(manager.run_dir(run_id), trial.mocap_file() if trial else None)
        return page(request, "run.html", current="live", trial=trial, kind="runs", key=run_id, run=run,
                    preset=PRESETS.get(run.preset), progress=manager.status_payload(run), **context)

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> JSONResponse:
        run = manager.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run")
        return JSONResponse(manager.status_payload(run), headers={"Cache-Control": "no-store"})

    @app.get("/files/{kind}/{key}/{filename}")
    def result_file(kind: str, key: str, filename: str) -> FileResponse:
        if kind == "showcase":
            trial = catalog.get(key)
            base = trial.showcase_dir if trial else None
        elif kind == "runs" and manager.get(key) is not None:
            base = manager.run_dir(key)
        else:
            base = None
        path = safe_file(base, filename) if base else None
        if path is None:
            raise HTTPException(404, "Not found")
        disposition = "inline" if filename.endswith((".mp4", ".json")) else "attachment"
        return FileResponse(path, media_type=RESULT_FILES[filename],
                            content_disposition_type=disposition, filename=f"{key}_{filename}")

    @app.get("/about", response_class=HTMLResponse)
    def about(request: Request) -> HTMLResponse:
        return page(request, "about.html", current="about")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        if request.url.path.startswith(("/api/", "/files/")):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return page(request, "message.html", status_code=exc.status_code, title=str(exc.detail),
                    message="", link="/")

    return app
