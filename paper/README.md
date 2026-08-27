# JSSR manuscript workspace

This directory is the writing workspace for the following target venue:

- Journal: *Journal of Safety Science and Resilience* (JSSR)
- Special issue: *Agent for Safety and Security: Complex Network Modeling, Substructure Analysis, Risk Discovery, and Resilience Assessment*
- Submission deadline: 28 February 2027
- Submission selection: `SI: Agent for Safety and Security`

## Working title

> Knowledge-Graph-Augmented and Evidence-Grounded LLM Agents for Risk Information Extraction in Low-Resource Safety Texts: Railway Accident Reports as a Primary Evaluation Scenario

The title deliberately focuses on risk information extraction. Risk-substructure discovery, cascading-risk analysis, and resilience assessment should not be added to the title until graph-level experiments support those claims.

## Files

- `elsarticle.zip`: original Elsevier template archive obtained from the remote repository.
- `elsarticle/manuscript.tex`: anonymous manuscript for double-blind review.
- `elsarticle/title-page.tex`: author and affiliation page uploaded separately.
- `elsarticle/references.bib`: BibTeX database.
- `EXPERIMENT_PLAN.md`: required experiments, publication gates, and schedule.

The extracted package documentation and LaTeX build products are ignored. The required class and numeric bibliography style are versioned with the manuscript.

## Build

```bash
cd paper/elsarticle
latexmk -pdf manuscript.tex
latexmk -pdf title-page.tex
```

Clean generated files with:

```bash
latexmk -C manuscript.tex
latexmk -C title-page.tex
```

## Submission requirements

- Keep the review manuscript anonymous because JSSR uses double-blind review.
- Upload the title page separately with all authors, affiliations, emails, and corresponding-author details.
- Use numbered references in square brackets.
- Upload editable source files and high-resolution figure files.
- Select the special-issue article type in Editorial Manager.
- Verify the current article processing charge before submission.

## Official references

- Special issue: https://www.keaipublishing.com/en/journals/journal-of-safety-science-and-resilience/call-for-papers/special-issue-on-agent-for-safety-and-security-complex-network-modeling-substructure-analysis-risk-discovery-and-resilience-assessment/
- Guide for Authors: https://www.keaipublishing.com/en/journals/journal-of-safety-science-and-resilience/guide-for-authors/
- Submission system: https://www.editorialmanager.com/JNLSSR/default.aspx
