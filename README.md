# ATF AI Challenge 2026

Monorepo for the ATF AI Challenge work. Every project keeps its own folder under
`atfaichallenge/`, and every project keeps its **full original commit history**.

```
atfaichallenge/
├── nomaapp-fastapi-server/   # FastAPI inference service for NomaApp (crop disease AI)
├── atf_ai_weekOne/           # Course: AI foundations — probability, LLMs, transformers, prompting
├── weekTwo/                  # Course: text processing, context windows, embeddings, RAG
├── weekThree/                # Course: tool use & agents, evals, guardrails, red teaming
└── weekFour/                 # Course: Streamlit UI, deployment, pitching, mobile reality
```

## What changed

This repository started life as the NomaApp FastAPI server. Its files now live in
`atfaichallenge/nomaapp-fastapi-server/` so that the four `techatatf` course
repositories can sit beside it as siblings.

| Project | Origin | Imported |
| --- | --- | --- |
| `nomaapp-fastapi-server` | this repo (fork of `Yusufiabichi/nomaapp-fastapi-server`) | — (moved in place) |
| `atf_ai_weekOne` | fork of `techatatf/weekOne` | 2026-11-13 |
| `weekTwo` | fork of `techatatf/weekTwo` | 2026-11-13 |
| `weekThree` | fork of `techatatf/weekThree` | 2026-11-13 |
| `weekFour` | fork of `techatatf/weekFour` | 2026-11-13 |

## Running the NomaApp API

```bash
cd atfaichallenge/nomaapp-fastapi-server
pip install -r requirements.txt
uvicorn main:app --reload
```

The service listens on `$PORT` (default `8000`); docs are served at `/docs`.

> Heads-up: `requirements.txt` is saved as UTF-16, which some tooling chokes on.
> Convert it to UTF-8 (`iconv -f UTF-16 -t UTF-8 requirements.txt`) if `pip install` complains.

## Deployment note

The Dockerfile was written for a repo-root build context, and the app is no
longer at the root. Build from the project folder instead:

```bash
docker build -t nomaapp-api atfaichallenge/nomaapp-fastapi-server
```

On Railway / Cloud Run, set the service's build context (root directory) to
`atfaichallenge/nomaapp-fastapi-server`, otherwise `COPY requirements.txt .`
will fail.

## Working with the imported repos

Each course repo was imported with `git subtree`, so its history, authors and
commit dates are preserved and remain easy to update.

Pull later changes from a course repo's `main` branch:

```bash
git subtree pull --prefix=atfaichallenge/weekTwo https://github.com/Umarabdul270/weekTwo.git main
```

Browse one project's history:

```bash
git log --oneline --follow -- atfaichallenge/weekTwo
```

Push changes made inside a subfolder back to its original repo:

```bash
git subtree push --prefix=atfaichallenge/weekTwo https://github.com/Umarabdul270/weekTwo.git main
```

Source repositories were left untouched — nothing was archived or deleted.
