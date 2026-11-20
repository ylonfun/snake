Set fso = CreateObject("Scripting.FileSystemObject")
Set ws = CreateObject("WScript.Shell")
strDir = fso.GetParentFolderName(WScript.ScriptFullName)
ws.CurrentDirectory = strDir
strPython = "C:\Users\yu320\AppData\Local\Doubao\User Data\sandbox_runtime\bases\772984f6fd3a8ae0ae7d2422cd7c2bfa\python\pythonw.exe"
ws.Run """" & strPython & """ """ & fso.BuildPath(strDir, "launcher.py") & """", 0, False
