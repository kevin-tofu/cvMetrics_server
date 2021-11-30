import os
import config
# from logconf import mylogger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import cvMetrics

app = FastAPI()
origins = [""]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

serverinfo= {
    "version": config.VERSION,
    "author": config.AUTHOR
}

@app.get("/")
def root():
    return serverinfo

app.include_router(cvMetrics.router)

if __name__ == "__main__":
    
    import uvicorn
    import argparse

    APP_PORT = config.APP_PORT

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-P', type=int, default=APP_PORT, help='port number for this server.')
    arg = parser.parse_args()

    uvicorn.run('server:app', host="0.0.0.0", port=arg.port)

