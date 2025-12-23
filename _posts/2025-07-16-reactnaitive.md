---
layout: post
title:  "리액트 네이티브 걸음마"
date:   2025-07-16 15:17
categories: MyStudy react
tag: react
---


1. node 설치
2. react native cli 설치
3. JDK 설치
    - 환경변수 설정
4. android studio 설치


# 환경 변수 설정 방법
1. 검색창에 "환경 변수" → "시스템 환경 변수 편집" 실행
2. 새 시스템 변수 추가
- 변수 이름: ANDROID_HOME<br>
변수 값: C:\Users\만수\AppData\Local\Android\Sdk
- 변수 이름: JAVA_HOME<BR>
변수 값: C:\Program Files\Java\jdk-17 : 자바 설치 경로

3. Path 편집 (시스템 변수)<br>
Path 항목 선택 → 편집 → 새로 만들기
```text
%ANDROID_HOME%\tools
%ANDROID_HOME%\build-tools
%ANDROID_HOME%\emulator
%ANDROID_HOME%\platform-tools
```
`adb --version`으로 확인

4. 사용자 변수 Path
```text
%ANDROID_HOME%\platform-tools
%ANDROID_HOME%\emulator
%ANDROID_HOME%\build-tools
%ANDROID_HOME%\tools
```


# 프로젝트


###  1. **신규 프로젝트 생성**
- 비주얼 스튜디오 코드에서 빈폴더를 연다. 
- 비주얼 스튜디오 코드 내에서 터미널을 키고 다음을 입력한다

```bash
npx @react-native-community/cli init 프로젝트이름
```


### 2. **프로젝트 폴더로 이동**

```bash
cd 프로젝트이름
```

### 3. **기존 프로젝트인 경우 (`package.json` 있는 경우)**

```bash
npm install
```
* `node_modules`을 설치하는 과정
* 이미 설치돼 있다면 생략해도 됨


### 4. **안드로이드 앱 실행**

```bash
npm run android
```

* `ANDROID_HOME`, `JAVA_HOME` 등이 제대로 설정돼 있어야 함
* `npm run android` 실행 전에 **에뮬레이터를 먼저 켜 두거나** 실제 기기를 연결해 둬야 함.
* 오래걸릴 수도 있으니 천천히 기다리기
* 완료되면 'Welcome to React Native' 화면의 애뮬레이터가 보인다

# 시행착오

### BUILD FAIL 
:app:configureCMakeDebug[arm64-v8a]

java 버전 17로 낮추기

