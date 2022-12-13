conda info --env

cd testpyt

conda create -n pyt python=3.8

=====

git --version

===

echo "# testpyt" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:guangyuli-uoe/testpyt.git
git push -u origin main