# Ajusta la tarjeta de red "Ethernet" (Intel I219-V) para recibir las imagenes de los sensores sin perder paquetes.
# Ejecutar en PowerShell COMO ADMINISTRADOR. La conexion de red se corta unos segundos al aplicar.

$nic = "Ethernet"

# Mas memoria para recibir paquetes (estaba en 256, el maximo es 2048): la tarjeta descarto ~195.000 paquetes
Set-NetAdapterAdvancedProperty -Name $nic -RegistryKeyword "*ReceiveBuffers" -RegistryValue 2048 -NoRestart

# Ahorro de energia que puede pausar la tarjeta o el enlace
Set-NetAdapterAdvancedProperty -Name $nic -RegistryKeyword "EEELinkAdvertisement" -RegistryValue 0 -NoRestart   # Ethernet con uso eficiente de energia
Set-NetAdapterAdvancedProperty -Name $nic -RegistryKeyword "*SelectiveSuspend" -RegistryValue 0 -NoRestart       # Suspension selectiva
Set-NetAdapterAdvancedProperty -Name $nic -RegistryKeyword "ULPMode" -RegistryValue 0 -NoRestart                 # Modo de energia ultrabaja

# Reinicia la tarjeta para aplicar los cambios
Restart-NetAdapter -Name $nic

Get-NetAdapterAdvancedProperty -Name $nic | Where-Object { $_.RegistryKeyword -in "*ReceiveBuffers", "EEELinkAdvertisement", "*SelectiveSuspend", "ULPMode" } |
    Select-Object DisplayName, DisplayValue | Format-Table -AutoSize
