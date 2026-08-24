// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title EmergencyNotificationService
 * @dev Dispatches high-priority emergency alerts and broadcasts on-chain events
 * to municipal response teams, police, and ambulances for critical traffic accidents.
 */
contract EmergencyNotificationService {
    address public admin;

    struct Responder {
        string department;
        string contactEndpoint;
        bool isAuthorized;
    }

    mapping(address => Responder) public responders;
    uint256 public totalAlertsDispatched;

    event EmergencyAlertDispatched(
        uint256 indexed incidentId,
        string ipfsHash,
        uint8 severity,
        string message,
        uint256 dispatchedAt
    );

    event ResponderAuthorized(address indexed responder, string department);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this operation");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    function registerResponder(
        address _responder,
        string memory _department,
        string memory _contactEndpoint
    ) external onlyAdmin {
        require(_responder != address(0), "Invalid responder address");
        responders[_responder] = Responder({
            department: _department,
            contactEndpoint: _contactEndpoint,
            isAuthorized: true
        });
        emit ResponderAuthorized(_responder, _department);
    }

    function notifyEmergency(
        uint256 _incidentId,
        string memory _ipfsHash,
        uint8 _severity,
        string memory _details
    ) external {
        totalAlertsDispatched++;
        emit EmergencyAlertDispatched(
            _incidentId,
            _ipfsHash,
            _severity,
            _details,
            block.timestamp
        );
    }
}
