def get_top_words(model, text, top_n=5):
    tfidf = model.named_steps["tfidf"]
    vector = tfidf.transform([text])

    feature_names = tfidf.get_feature_names_out()
    scores = vector.toarray()[0]

    top_indices = scores.argsort()[-top_n:][::-1]

    return [feature_names[i] for i in top_indices]