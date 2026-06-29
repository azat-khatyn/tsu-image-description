# UI-контейнер: nginx отдаёт статику и проксирует API в backend.
# Контекст сборки — корень репозитория (нужен app/ui/index.html).
FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY app/ui/index.html /usr/share/nginx/html/index.html

EXPOSE 80
