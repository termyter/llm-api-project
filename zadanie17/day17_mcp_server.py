"""
День 17 — MCP Server: GitHub API Tools
=======================================
MCP-сервер, оборачивающий GitHub REST API.

Инструменты:
  get_repo_info      — основная информация о репозитории
  get_latest_commits — последние коммиты
  list_open_issues   — открытые задачи

Агент подключается через stdio transport, вызывает инструменты
и использует результаты для анализа репозиториев без дополнительного CLI.
"""

import httpx
from mcp.server.fastmcp import FastMCP

# ── Сервер ────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="GitHub Tools",
    instructions=(
        "Инструменты для работы с GitHub REST API. "
        "Возвращают информацию о репозиториях, коммитах и задачах. "
        "Не требуют токена для публичных репозиториев."
    ),
)

_API = "https://api.github.com"
_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ── Инструменты ───────────────────────────────────────────────────────────────

@mcp.tool()
def get_repo_info(owner: str, repo: str) -> dict:
    """
    Возвращает основную информацию о GitHub-репозитории.

    Параметры:
      owner — логин владельца репозитория (пользователь или организация)
      repo  — название репозитория

    Возвращает словарь:
      name, description, stars, forks, language, open_issues, url,
      created_at, updated_at
    """
    r = httpx.get(f"{_API}/repos/{owner}/{repo}", headers=_HEADERS, timeout=10)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:300]}
    d = r.json()
    return {
        "name":        d["full_name"],
        "description": d.get("description") or "—",
        "stars":       d["stargazers_count"],
        "forks":       d["forks_count"],
        "language":    d.get("language") or "—",
        "open_issues": d["open_issues_count"],
        "url":         d["html_url"],
        "created_at":  d["created_at"][:10],
        "updated_at":  d["updated_at"][:10],
    }


@mcp.tool()
def get_latest_commits(owner: str, repo: str, limit: int = 5) -> list:
    """
    Возвращает список последних коммитов в репозитории.

    Параметры:
      owner — логин владельца репозитория
      repo  — название репозитория
      limit — максимальное количество коммитов (по умолчанию 5, макс. 20)

    Возвращает список словарей: sha (7 символов), message (первая строка),
    author, date (YYYY-MM-DD).
    """
    limit = min(limit, 20)
    r = httpx.get(
        f"{_API}/repos/{owner}/{repo}/commits",
        headers=_HEADERS,
        params={"per_page": limit},
        timeout=10,
    )
    if r.status_code != 200:
        return [{"error": f"HTTP {r.status_code}"}]
    return [
        {
            "sha":     c["sha"][:7],
            "message": c["commit"]["message"].split("\n")[0][:80],
            "author":  c["commit"]["author"]["name"],
            "date":    c["commit"]["author"]["date"][:10],
        }
        for c in r.json()
    ]


@mcp.tool()
def list_open_issues(owner: str, repo: str, limit: int = 5) -> list:
    """
    Возвращает список открытых issue в репозитории (без pull request).

    Параметры:
      owner — логин владельца репозитория
      repo  — название репозитория
      limit — максимальное количество issue (по умолчанию 5, макс. 20)

    Возвращает список словарей: number, title, url, created_at, labels.
    Если открытых задач нет — возвращает пустой список [].
    """
    limit = min(limit, 20)
    r = httpx.get(
        f"{_API}/repos/{owner}/{repo}/issues",
        headers=_HEADERS,
        params={"state": "open", "per_page": limit},
        timeout=10,
    )
    if r.status_code != 200:
        return [{"error": f"HTTP {r.status_code}"}]
    return [
        {
            "number":     i["number"],
            "title":      i["title"],
            "url":        i["html_url"],
            "created_at": i["created_at"][:10],
            "labels":     [lb["name"] for lb in i.get("labels", [])],
        }
        for i in r.json()
        if "pull_request" not in i  # исключаем PR из выдачи
    ][:limit]


# ── Запуск ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
