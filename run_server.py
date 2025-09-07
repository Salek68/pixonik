import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # مسیر فایل و اسم app
        host="127.0.0.1",
        port=8000,
        reload=True,   # برای ری‌لود خودکار هنگام تغییر کد
    )
