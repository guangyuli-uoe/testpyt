conda info --env

cd testpyt

conda create -n pyt python=3.8

=====

git --version

====

git init

cat ~/.ssh/id_rsa.pub

git remote add origin git@github.com:guangyuli-uoe/testpyt.git

git add readme.txt
git commit -m '1213'

git branch -m master main

git push -u origin main
===

echo "# testpyt" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:guangyuli-uoe/testpyt.git
git push -u origin main


'''
        1.19
        撤回add: git reset HEAD
        
'''
normal:
    git add .
    git commit -m ''
    git push