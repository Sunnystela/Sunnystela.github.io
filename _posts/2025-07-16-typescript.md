---
layout: post
title:  "TypeScript 기초 입력 방법"
date:   2025-07-16 10:00
categories:  MyStudy react
tag: react
---

# TypeScript 시작하기


## 0. Node.js 설치 (tsc를 쓰기 위해 필요함)

* [https://nodejs.org/](https://nodejs.org/) 접속 → **LTS 버전 다운로드 & 설치**
* 설치 후, **cmd** 열어서 아래 명령 입력:

  ```bash
  node -v
  npm -v
  ```

  버전 숫자가 나오면 설치 성공.


## 1. VS Code에서 프로젝트 폴더 만들기

(1) VS Code 열기

* `파일 > 폴더 열기` → 새 폴더 하나 만들고 엽니다. 

(2) VS Code에서 터미널 열기

* 상단 메뉴 → `터미널 > 새 터미널` 클릭 → **하단에 터미널(cmd 창)** 생김


## 2. TypeScript 환경 설정

(VS Code 터미널에 아래 명령 순서대로 입력)

```bash
npm init -y                     # package.json 자동 생성
npm install typescript --save-dev    # 타입스크립트 설치
npx tsc --init                  # tsconfig.json 생성 (설정파일)
```

> 생기는 파일
>
> * `package.json`
> * `tsconfig.json`
> * `node_modules` 폴더



## 3. 코드 파일 만들기

VS Code에서 `src/index.ts` 파일을 새로 만들기

1. 왼쪽 탐색기에서 `src` 폴더 생성
2. 그 안에 `index.ts` 파일 생성
3. 다음 코드 입력:

```ts
function greet(name: string): void {
  console.log(`안녕, ${name}!`);
}

greet("만수");
```



## 4. tsconfig.json 설정 조금 수정

`tsconfig.json` 파일에서 아래 2줄 주석 제거 (Ctrl+F로 찾아서)
주석 제거가 코드를 지우는게 아니라 주석 해제<br>
dist, src 입력해준다.

```json
"outDir": "./dist"
"rootDir": "./src"
```



## 5. TypeScript 코드 컴파일하고 실행

VS Code 하단 터미널에 입력:

```bash
npx tsc    # index.ts → index.js 변환됨
```

→ `dist/index.js` 생김

```bash
node dist/index.js    # 실행 결과: "안녕, 만수!"
```



## 6. 자동 컴파일 (선택)

터미널에 아래 입력해서 실시간으로 `.ts` → `.js` 자동 변환되게 하기:

```bash
npx tsc -w
```
