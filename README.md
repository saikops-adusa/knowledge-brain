# knowledge-brain

Upload your PDF presentations and ask questions against their content.

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   streamlit run app.py
   ```

3. In the app:
   - Upload one or more presentation PDFs
   - Ask a question in the chatbot input
   - The app returns the most relevant extracted passages from your uploaded documents

## Tests

Run focused unit tests:

```bash
python -m unittest discover -s tests
```
