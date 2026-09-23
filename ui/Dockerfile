# Operator UI v2 SPA — multi-stage build.
# Stage 1: build the static bundle with Vite.
FROM node:25-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: serve the static bundle with nginx; the API base is injected at
# container start (see docker-entrypoint.sh) so one image runs against any
# indexer. No bearer token is ever baked into the image.
FROM nginx:alpine AS serve
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
# nginx:alpine runs /docker-entrypoint.d/*.sh before starting nginx, so the
# config-generator slots in there; the default CMD (nginx -g 'daemon off;') stays.
COPY docker-entrypoint.sh /docker-entrypoint.d/40-treeloom-config.sh
RUN chmod +x /docker-entrypoint.d/40-treeloom-config.sh
EXPOSE 80
