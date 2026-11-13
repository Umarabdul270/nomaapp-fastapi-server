# ATF AI Challenge 2026 — course repos

The ATF AI Challenge 2026 course material, collected in one place. Each folder
is a fork of a `techatatf` repository, imported with `git subtree` so its full
commit history, authors and dates are preserved.

| Folder | Fork of | Contents |
| --- | --- | --- |
| `atf_ai_weekOne/` | `techatatf/weekOne` | AI foundations: probability & neural networks, LLM architecture, transformer attention, prompting, API integration |
| `weekTwo/` | `techatatf/weekTwo` | String surgery / text processing, context window management, embeddings, RAG |
| `weekThree/` | `techatatf/weekThree` | Tool use & AI agents, evaluations & testing, guardrails & safety, red teaming |
| `weekFour/` | `techatatf/weekFour` | Streamlit frontend UI, deployment & hosting, pitching & storytelling, mobile reality |

> The NomaApp FastAPI server intentionally lives at the **repository root**, not
> in this folder — it is the product repo, not part of the course material.

## Working with these folders

Each folder was imported with `git subtree`, so upstream updates stay easy.

Pull the latest changes for one course repo:

```bash
git subtree pull --prefix=atfaichallenge/weekTwo https://github.com/Umarabdul270/weekTwo.git main
```

View one folder's history:

```bash
git log --oneline -- atfaichallenge/weekTwo
```

Push changes made inside a folder back to its original repo:

```bash
git subtree push --prefix=atfaichallenge/weekTwo https://github.com/Umarabdul270/weekTwo.git main
```

The original `techatatf` forks were left untouched — nothing was archived or deleted.
