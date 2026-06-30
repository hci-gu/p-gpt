# PsychiatrictGPT (P-GPT)




## Installation & Setup


### Backend

Navigate to `p-gpt/backend/` for non-cuda compatible device, run
``` bash
uv sync --extra cpu
```
For CUDA compatible device, run
``` bash
uv sync --extra cuda
```

To spin up the FastAPI server
``` bash
fastapi dev
```

Server will be [http://127.0.0.1:8000](http://127.0.0.1:8000) and the docs can easily be accesed from [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)


## Production

### Backend

For production instead opt for starting the server using
``` bash
fastapi run app.py
``` 
or for more specific host and port control
``` bash
uvicorn main:app --host 0.0.0.0 --port 80
```