import uvicorn


if __name__ == "__main__":
    uvicorn.run('app:app', host='0.0.0.0', port=9001, log_level="info",reload=True )
    # uvicorn.run(app, host='0.0.0.0', port=9001, log_level="info", reload=True)
