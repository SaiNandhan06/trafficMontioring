// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TrafficIncidentRegistry
 * @dev Decentralized immutable registry for UAV drone traffic monitoring,
 * incident reporting, IPFS evidence verification, and incident lifecycle management.
 */
contract TrafficIncidentRegistry {
    address public owner;

    enum Severity { LOW, MEDIUM, HIGH, CRITICAL }
    enum IncidentStatus { REPORTED, ESCALATED, UNDER_INVESTIGATION, RESOLVED }

    struct DroneInfo {
        string droneId;
        string metadata;
        bool isActive;
        uint256 registeredAt;
    }

    struct IncidentRecord {
        uint256 incidentId;
        string ipfsHash;         // CID pointing to tamper-proof image snapshot and metadata
        string incidentType;     // SPEEDING, COLLISION_ACCIDENT, LANE_VIOLATION, TRAFFIC_CONGESTION
        Severity severity;
        int256 latitude;         // Scaled by 1e6 for fixed-point representation
        int256 longitude;        // Scaled by 1e6 for fixed-point representation
        uint256 timestamp;       // Unix timestamp
        address reportingDrone;  // Address of authorized UAV edge node
        IncidentStatus status;
        string resolutionNotes;
        uint256 resolvedAt;
    }

    // Mappings
    mapping(address => DroneInfo) public registeredDrones;
    mapping(uint256 => IncidentRecord) public incidents;
    uint256 public incidentCount;

    // Events
    event DroneRegistered(address indexed droneAddress, string droneId, uint256 registeredAt);
    event DroneDeactivated(address indexed droneAddress);
    event IncidentReported(
        uint256 indexed incidentId,
        string ipfsHash,
        string incidentType,
        Severity severity,
        int256 latitude,
        int256 longitude,
        uint256 timestamp,
        address indexed reportingDrone
    );
    event IncidentEscalated(uint256 indexed incidentId, string reason, uint256 escalatedAt);
    event IncidentResolved(uint256 indexed incidentId, string resolutionNotes, uint256 resolvedAt);

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only registry owner can execute this action");
        _;
    }

    modifier onlyActiveDrone() {
        require(
            registeredDrones[msg.sender].isActive || msg.sender == owner,
            "Caller is not an active authorized UAV drone"
        );
        _;
    }

    constructor() {
        owner = msg.sender;
        // Register deployer as default authorized drone for testing
        registeredDrones[msg.sender] = DroneInfo({
            droneId: "UAV-SYSTEM-ROOT",
            metadata: "Initial Deployer UAV",
            isActive: true,
            registeredAt: block.timestamp
        });
    }

    /**
     * @notice Registers a new UAV edge drone node in the registry.
     */
    function registerDrone(
        address _droneAddress,
        string memory _droneId,
        string memory _metadata
    ) external onlyOwner {
        require(_droneAddress != address(0), "Invalid drone address");
        registeredDrones[_droneAddress] = DroneInfo({
            droneId: _droneId,
            metadata: _metadata,
            isActive: true,
            registeredAt: block.timestamp
        });
        emit DroneRegistered(_droneAddress, _droneId, block.timestamp);
    }

    /**
     * @notice Deactivates an existing UAV drone node.
     */
    function deactivateDrone(address _droneAddress) external onlyOwner {
        require(registeredDrones[_droneAddress].isActive, "Drone is not currently active");
        registeredDrones[_droneAddress].isActive = false;
        emit DroneDeactivated(_droneAddress);
    }

    /**
     * @notice Reports a verified traffic incident detected by an authorized UAV.
     */
    function reportIncident(
        string memory _ipfsHash,
        string memory _incidentType,
        Severity _severity,
        int256 _latitude,
        int256 _longitude,
        uint256 _timestamp
    ) external onlyActiveDrone returns (uint256) {
        require(bytes(_ipfsHash).length > 0, "IPFS hash required");

        incidentCount++;
        uint256 currentId = incidentCount;

        incidents[currentId] = IncidentRecord({
            incidentId: currentId,
            ipfsHash: _ipfsHash,
            incidentType: _incidentType,
            severity: _severity,
            latitude: _latitude,
            longitude: _longitude,
            timestamp: _timestamp > 0 ? _timestamp : block.timestamp,
            reportingDrone: msg.sender,
            status: IncidentStatus.REPORTED,
            resolutionNotes: "",
            resolvedAt: 0
        });

        emit IncidentReported(
            currentId,
            _ipfsHash,
            _incidentType,
            _severity,
            _latitude,
            _longitude,
            _timestamp > 0 ? _timestamp : block.timestamp,
            msg.sender
        );

        return currentId;
    }

    /**
     * @notice Escalates an incident to municipal or emergency authorities.
     */
    function escalateIncident(uint256 _incidentId, string memory _reason) external {
        require(_incidentId > 0 && _incidentId <= incidentCount, "Incident does not exist");
        IncidentRecord storage inc = incidents[_incidentId];
        inc.status = IncidentStatus.ESCALATED;
        emit IncidentEscalated(_incidentId, _reason, block.timestamp);
    }

    /**
     * @notice Resolves an incident once cleared by emergency/traffic responders.
     */
    function resolveIncident(uint256 _incidentId, string memory _notes) external onlyOwner {
        require(_incidentId > 0 && _incidentId <= incidentCount, "Incident does not exist");
        IncidentRecord storage inc = incidents[_incidentId];
        inc.status = IncidentStatus.RESOLVED;
        inc.resolutionNotes = _notes;
        inc.resolvedAt = block.timestamp;
        emit IncidentResolved(_incidentId, _notes, block.timestamp);
    }

    /**
     * @notice Fetches complete details of an incident.
     */
    function getIncident(uint256 _incidentId) external view returns (IncidentRecord memory) {
        require(_incidentId > 0 && _incidentId <= incidentCount, "Incident does not exist");
        return incidents[_incidentId];
    }
}
