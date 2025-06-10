#!/usr/bin/env python3

import os
import asyncio
import pytest
from hypha_rpc import connect_to_server

@pytest.mark.asyncio
async def test_connection():
    token = os.environ.get('SQUID_WORKSPACE_TOKEN')
    if not token:
        print('❌ No SQUID_WORKSPACE_TOKEN found in environment')
        return False
    
    print('🔗 Attempting to connect to Hypha server...')
    try:
        server = await connect_to_server({
            'server_url': 'https://hypha.aicell.io',
            'token': token,
            'workspace': 'squid-control',
            'ping_interval': None
        })
        print('✅ Successfully connected to server')
        print(f'📊 Server workspace: {server.config.workspace}')
        return True
    except Exception as e:
        print(f'❌ Connection failed: {e}')
        return False

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    exit(0 if result else 1) 