@echo off
echo Обновление setuptools и pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Установка зависимостей...
python -m pip install -r requirements.txt

echo.
echo Установка завершена!
pause


