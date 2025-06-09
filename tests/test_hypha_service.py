import pytest
from unittest.mock import patch, AsyncMock
import os

from start_hypha_service import Microscope

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Condition to run local tests. Set HYPHA_TEST_LOCAL=1 in your environment to run them.
run_local = os.environ.get("HYPHA_TEST_LOCAL") == "1"

@patch('start_hypha_service.connect_to_server', new_callable=AsyncMock)
async def test_connect_to_similarity_search_service_remote(mock_connect_to_server):
    """Test connect_to_similarity_search_service with is_local=False."""
    # Arrange
    mock_server = AsyncMock()
    mock_service = AsyncMock()
    mock_server.get_service.return_value = mock_service
    mock_connect_to_server.return_value = mock_server

    # We need to mock SquidController as it's initialized in Microscope.__init__
    with patch('start_hypha_service.SquidController') as mock_squid_controller:
        mock_squid_controller.return_value = AsyncMock()
        microscope = Microscope(is_simulation=True, is_local=False)
    
        # Act
        svc = await microscope.connect_to_similarity_search_service()

        # Assert
        assert svc is mock_service
        mock_connect_to_server.assert_awaited_once()
        call_args = mock_connect_to_server.call_args[0][0]
        assert call_args['server_url'] == 'https://hypha.aicell.io'
        assert call_args['workspace'] == 'agent-lens'
        mock_server.get_service.assert_awaited_once_with("image-text-similarity-search")

@pytest.mark.skipif(not run_local, reason="Set HYPHA_TEST_LOCAL=1 to run local tests")
@patch('start_hypha_service.connect_to_server', new_callable=AsyncMock)
async def test_connect_to_similarity_search_service_local(mock_connect_to_server):
    """Test connect_to_similarity_search_service with is_local=True."""
    # Arrange
    mock_server = AsyncMock()
    mock_service = AsyncMock()
    mock_server.get_service.return_value = mock_service
    mock_connect_to_server.return_value = mock_server

    with patch('start_hypha_service.SquidController') as mock_squid_controller:
        mock_squid_controller.return_value = AsyncMock()
        microscope = Microscope(is_simulation=True, is_local=True)

        # Act
        svc = await microscope.connect_to_similarity_search_service()

        # Assert
        assert svc is mock_service
        mock_connect_to_server.assert_awaited_once()
        call_args = mock_connect_to_server.call_args[0][0]
        assert call_args['server_url'] == 'http://192.168.2.1:9527'
        mock_server.get_service.assert_awaited_once_with("image-text-similarity-search") 