#!/bin/bash
# Monitor training progress

echo "Monitoring PINN Training..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo "================================"
echo ""

tail -f /tmp/claude/-root-Documents-GallingModel/tasks/b2eb539.output
