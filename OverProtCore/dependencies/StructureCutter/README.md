# StructureCutter

**StructureCutter** is a utility for batch loading/downloading protein structures in mmCIF format and extracting specified regions (domains) from these structures.

## Dependencies

You must have .NET Core 6.0 runtime installed.

Note: I couldn't install it via `apt` or `snap` on Ubuntu22.04 (tried on 15 May 2022) so that it would work.
I installed it like this:

```sh
wget https://dotnet.microsoft.com/download/dotnet/scripts/v1/dotnet-install.sh
sudo bash dotnet-install.sh -c 6.0 --install-dir /opt/dotnet60
sudo ln -s /opt/dotnet60/dotnet /usr/local/bin/dotnet60
sudo ln -s dotnet60 /usr/local/bin/dotnet
dotnet --info
# might need a reboot here (especially if build is not working)
```

## Build

```sh
dotnet build  # Debug
dotnet build -c Release  
```

## Execution

```sh
dotnet ./bin/Release/net6.0/StructureCutter.dll --help
```
