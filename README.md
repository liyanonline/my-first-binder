# my-first-binder

python3 -m pip install pandas matplotlib sklearn statsmodels mord seaborn dmba mlxtend surprise

C:\Users\liyan\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\Scripts

R -e "install.packages(c('forecast', 'DiscriMiner', 'arules','neuralnet', 'gplots', 'reshape','GGally','treemap','igraph','ggmap'),repos = 'http://cran.us.r-project.org')"

RScript Ch10.R

R -e "source('Ch10.R')"

choco install vcredist140

https://chocolatey.org/install

Install Chocolatey for Individual Use:
First, ensure that you are using an administrative shell - you can also install as a non-admin, check out Non-Administrative Installation.
Install with powershell.exe

📝 NOTE: Please inspect https://community.chocolatey.org/install.ps1 prior to running any of these scripts to ensure safety. We already know it's safe, but you should verify the security and contents of any script from the internet you are not familiar with. All of these scripts download a remote PowerShell script and execute it on your machine. We take security very seriously. Learn more about our security protocols.
With PowerShell, you must ensure Get-ExecutionPolicy is not Restricted. We suggest using Bypass to bypass the policy to get things installed or AllSigned for quite a bit more security.

Run Get-ExecutionPolicy. If it returns Restricted, then run Set-ExecutionPolicy AllSigned or Set-ExecutionPolicy Bypass -Scope Process.
Now run the following command:

Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
Paste the copied text into your shell and press Enter.
Wait a few seconds for the command to complete.
If you don't see any errors, you are ready to use Chocolatey! Type choco or choco -? now, or see Getting Started for usage instructions.

