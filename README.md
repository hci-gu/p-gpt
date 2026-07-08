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

### Frontend

Navigate to `p-gpt/frontend/`, install dependencies, then run the Vite dev server.
``` bash
pnpm install
```

Create a `.env.local` file in `p-gpt/frontend/` with the backend API endpoint.
``` env
VITE_API_ENDPOINT=http://127.0.0.1:8000
```

To spin up the frontend
``` bash
pnpm dev
```

Frontend will be [http://localhost:5173](http://localhost:5173)

### PocketBase

Navigate to `p-gpt/pocketbase/`, then run the PocketBase server.
``` bash
go run . serve
```

PocketBase will be [http://127.0.0.1:8090](http://127.0.0.1:8090) and the admin dashboard will be [http://127.0.0.1:8090/_/](http://127.0.0.1:8090/_/)


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

### Frontend

For production, build the frontend from `p-gpt/frontend/` using
``` bash
pnpm build
```

To preview the production build locally
``` bash
pnpm preview
```

### PocketBase

For production, build the PocketBase app from `p-gpt/pocketbase/` using
``` bash
go build
```

Then start the compiled binary.
``` bash
./app serve
```
On Windows, run `./app.exe serve` instead.
