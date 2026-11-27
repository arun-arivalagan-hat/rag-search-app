"""Measure whether search is good, so every change is judged by numbers.

    qrels.py     load the labelled questions (data/eval/*.json)
    metrics.py   Hit@k, Precision@k, Recall@k, reciprocal rank
    runner.py    run a search method over all questions and summarise
    tuning.py    try many settings, with a held-out check against overfitting
    answers.py   answer-level check: does a top chunk contain the expected phrase?
"""
