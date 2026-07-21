# Reply to Xuyan Ye (English version)

> Translated from the confirmed Chinese draft. Do not commit this file.

---

Dear Xuyan,

Sorry for the late reply — I was attending the ICML conference and things have been quite busy since I got back.

Thank you for reading the paper and the code so carefully — your understanding of the implementation details is accurate, and both questions get at important points. Let me respond point by point.

## On Question 1 (cost and role of per-step trajectory summarization)

**1.1 Was per-step summarization used in the reported experiments? Did we try incremental or cached summaries?**

Yes — the reported Jericho experiments use exactly this configuration: at each step, a structured summary is generated from the full trajectory so far; the summary text goes into the action-generation prompt and also serves as the retrieval query (it does double duty). Your observation that the cumulative summarization input grows roughly quadratically with the episode horizon is correct. In practice, however, this is not a real issue on Jericho: the per-turn text in these text-adventure games is very short, so even feeding the full trajectory at every step keeps the absolute cost small. For that reason we did not add incremental summarization on Jericho — re-summarizing from the full history is simpler and gives better summaries, since the summarizer always sees the global picture.

The cached scheme you propose is in fact exactly what the WebArena code does: there, each step is summarized once, in isolation (`generate_single_step_summary`, cached in `step_summaries`), and the trajectory summary is the concatenation of the cached per-step summaries, so the total summarization input is O(T). The two implementations were chosen according to per-step text length: Jericho steps are short, so we use full re-summarization; a single WebArena AXTree observation runs to thousands of tokens, so incremental caching is necessary there.

**1.2 Latency / token / monetary cost relative to the base agent**

The total costs reported in the paper and rebuttal (e.g., ~$200 for the larger-budget WebArena experiment) are measured figures that include all LLM calls, summarization included. You are right, though, that the per-step overhead breakdown we provided to reviewers (retrieval 15–47 ms, advantage estimation <0.02 ms) covers only retrieval and advantage computation and does not itemize the summarization call — roughly speaking, per-step summarization adds about one extra LLM call per step, of a similar scale to the action-generation call, and it is the largest cost item in the pipeline besides action generation itself. We have not published a token/latency breakdown by call type.

**1.3 Separating the benefit of summarization from that of the advantage correction**

The closest evidence we have is **Table 8** (Prompt Update vs. Logit Update): both arms keep the per-step summaries and use exactly the same retrieved experiences; the only difference is the injection channel (appending the retrieved experiences to the prompt vs. applying them to the logits). Logit Update still wins (Admin 52.31 vs. 49.46, Reddit 57.64 vs. 53.02). Since the summary is common to both arms, this comparison shows that the contribution of the logit update is independent of the summarization.

## On Question 2 (complexity and generality of the retrieval pipeline)

**2.1 How much worse is a simpler retriever?**

We did try plain embedding top-k during development. Our conclusion: it mainly depends on the task, and Jericho happens to be the kind of environment that embedding retrieval handles worst. Many rooms in Jericho have nearly identical text descriptions that differ by only a few words, yet correspond to entirely different game states and returns — in embedding space, the cosine similarities among such states are all very high and nearly indistinguishable. The failure mode of pure embedding retrieval here is not missing relevant matches, but flooding the top-ranked matches with false positives; and a wrong match injects irrelevant historical returns into the advantage estimate, directly corrupting the action distribution. That is exactly why the lexical components exist: the unigram/4-gram Jaccard reranking picks up precisely those few-word differences (n-gram overlap is sensitive to local lexical differences that embeddings smooth away), and the dynamic threshold and the special handling of very-high-similarity matches serve the same purpose — controlling the false-positive rate of return transfer.

Conversely, in environments where states are naturally well separated (large textual differences between scenes, embeddings far apart), we agree that plain embedding top-k would be sufficient, and the lexical reranking becomes an optional safeguard. So we view these components as conservative protections against near-duplicate state spaces, rather than universal requirements. As related evidence, the unified-pipeline experiment in our rebuttal — which uses one identical state representation, retrieval procedure, and evaluator for both benchmarks — performs almost the same as the environment-specific versions (Jericho 25.9→24.7, WebArena 46.98→46.01) while still outperforming all baselines, indicating that the gains come mainly from the advantage estimation and logit update mechanism itself.

**2.2 Is environment-specific matching essential, or an implementation choice?**

Our view is the latter. The mechanism itself only requires two things: (i) a state-similarity function reliable enough for return transfer, and (ii) step-level reward signals. Designs like URL normalization are simply cheap shortcuts we found (they capture page type without an extra LLM call), not components of the framework — the unified-pipeline experiment is direct evidence for this. In our opinion, the more general direction is to use the model's hidden-state embeddings as dense state representations, combined with ANN retrieval (e.g., Faiss); this would free state matching from text engineering altogether and is part of our future work.

Thank you again for reading our work so carefully — we'd be glad to stay in touch, and we welcome opportunities to exchange ideas or collaborate down the road. Best of luck with your internship!

Best regards,
Yibo
