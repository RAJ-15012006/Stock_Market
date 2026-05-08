import React, { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Sphere, MeshDistortMaterial, Points, PointMaterial } from '@react-three/drei'
import * as THREE from 'three'

function RotatingGlobe() {
  const meshRef = useRef()
  
  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    meshRef.current.rotation.y = t * 0.1
    meshRef.current.rotation.x = Math.sin(t * 0.2) * 0.1
  })

  return (
    <group ref={meshRef}>
      {/* Central Holographic Sphere */}
      <Sphere args={[1, 64, 64]}>
        <MeshDistortMaterial
          color="#00f2ff"
          attach="material"
          distort={0.3}
          speed={2}
          roughness={0}
          emissive="#00f2ff"
          emissiveIntensity={0.5}
          transparent
          opacity={0.6}
        />
      </Sphere>
      
      {/* Atmosphere/Glow */}
      <Sphere args={[1.2, 64, 64]}>
        <meshBasicMaterial color="#00f2ff" wireframe transparent opacity={0.1} />
      </Sphere>

      {/* Market Data Points (Random Stars for demo) */}
      <Stars />
    </group>
  )
}

function Stars() {
  const count = 500
  const positions = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 10
    positions[i * 3 + 1] = (Math.random() - 0.5) * 10
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10
  }

  return (
    <Points positions={positions}>
      <PointMaterial
        transparent
        color="#00f2ff"
        size={0.05}
        sizeAttenuation={true}
        depthWrite={false}
      />
    </Points>
  )
}

export default function MarketGlobe() {
  return (
    <div className="w-full h-full absolute top-0 left-0 -z-10 bg-slate-950">
      <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <RotatingGlobe />
        <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
      </Canvas>
    </div>
  )
}
