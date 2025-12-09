## Summary

RAG implementation based on https://www.anthropic.com/engineering/contextual-retrieval

## Setting up the project

- Copy `env.example` to `.env`
- Install requirements with `pip`
```
$ pip install -r requirements.txt
```

## Run migrations
- Migrate with alembic
```
$ alembic upgrade head
```

# Run
- Store the docs
```
$ python main.py --process-docs
```
- Ask
```
$ python main.py --ask "question"
```