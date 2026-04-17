import os
import subprocess

vault_path = os.path.join(os.getcwd(), "Vault")

print("🔒 Engaging cloak...")
# Add System and Hidden attributes
subprocess.run(["attrib", "+h", "+s", vault_path])
print("👻 Vault is now completely hidden from the OS.")
