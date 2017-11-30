mysql -h <endpoint> -P 3306 -u <mymasteruser> -p

To Connect to MySQL on AWS: mysql -h nflstats.czvmrlxwom4l.us-east-2.rds.amazonaws.com -P 3306 -u root -p

Security group with access: sg-811264e9, LSU Inbound connections


Max APT External IP: 68.227.129.187


to convert .ppk to .pem: puttygen AdministratorAccount.ppk -O private-openssh -o AdministratorAccount.pem
- to set permissions: chmod 400 AdministratorAccount.pem

to connect from Linux: ssh -i AdministratorAccount.pem ec2-user@ec2-18-220-160-6.us-east-2.compute.amazonaws.com

or maybe: ssh -i AdministratorAccount.pem ec2-user@ff-ai.com


LSU public IPs that we know of: 
- 167.96.46.188/32
- 130.39.203.204/32
- 167.96.63.18/32
- 167.96.85.142/32
- 173.244.44.77/32
- 167.96.83.173/32
- 