@echo off
git fetch upstream
git checkout main
git merge upstream/main
git checkout Eric