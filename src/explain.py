def get_top_words(model, text, top_n=5):
    # Apply cleaning (same as pipeline)
    clean_text = model.named_steps["clean"].transform([text])

    # TF-IDF transform
    tfidf = model.named_steps["tfidf"]
    vector = tfidf.transform(clean_text)

    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    scores = vector.toarray()[0]

    # Top words
    top_indices = scores.argsort()[-top_n:][::-1]

    return [feature_names[i] for i in top_indices]