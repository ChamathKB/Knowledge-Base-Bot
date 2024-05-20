FROM node:20-bookworm-slim as build-step

WORKDIR /app

ENV PATH /app/node_modules/.bin:$PATH

COPY frontend/package.json  ./

RUN npm install

COPY ./frontend/ ./

RUN npm run build
