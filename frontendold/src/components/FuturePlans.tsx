import React from 'react';
import { 
  Brain, 
  Globe, 
  Zap, 
  Rocket, 
  Shield, 
  TrendingUp,
  Users,
  Factory,
  Leaf,
  Target,
  Award,
  Lightbulb
} from 'lucide-react';

export function FuturePlans() {
  const roadmapItems = [
    {
      phase: 'Q2 2025',
      title: 'AI-Powered Matching Engine',
      description: 'Advanced machine learning algorithms for precise material and partner matching',
      icon: Brain,
      status: 'in-progress',
      features: [
        'Real-time compatibility scoring',
        'Predictive demand forecasting',
        'Automated partnership suggestions',
        'Smart logistics optimization'
      ]
    },
    {
      phase: 'Q3 2025',
      title: 'Global Expansion',
      description: 'Expanding to 50+ countries with localized support and regulations',
      icon: Globe,
      status: 'planned',
      features: [
        'Multi-language support',
        'Regional compliance tools',
        'Local partnership networks',
        'Currency and measurement conversions'
      ]
    },
    {
      phase: 'Q4 2025',
      title: 'Blockchain Integration',
      description: 'Transparent supply chain tracking and verified sustainability metrics',
      icon: Shield,
      status: 'planned',
      features: [
        'Immutable transaction records',
        'Carbon footprint tracking',
        'Verified sustainability certificates',
        'Smart contract automation'
      ]
    },
    {
      phase: 'Q1 2026',
      title: 'IoT & Real-time Monitoring',
      description: 'Connected sensors for real-time material quality and quantity tracking',
      icon: Zap,
      status: 'research',
      features: [
        'Smart waste bin sensors',
        'Quality monitoring devices',
        'Automated inventory updates',
        'Predictive maintenance alerts'
      ]
    }
  ];

  const visionAreas = [
    {
      title: 'Zero Waste Economy',
      description: 'Creating a world where every material finds its perfect second life',
      icon: Leaf,
      metrics: ['95% waste diversion rate', '1M+ tons redirected annually']
    },
    {
      title: 'Global Network',
      description: 'Connecting 100,000+ organizations across all continents',
      icon: Users,
      metrics: ['100K+ active users', '200+ countries covered']
    },
    {
      title: 'Industry Leadership',
      description: 'Setting the standard for circular economy platforms worldwide',
      icon: Award,
      metrics: ['#1 platform globally', 'Industry partnerships']
    },
    {
      title: 'Innovation Hub',
      description: 'Fostering breakthrough technologies for sustainable resource management',
      icon: Lightbulb,
      metrics: ['50+ patents filed', '100+ research partnerships']
    }
  ];

  return (
    <div className="bg-slate-900 py-24">
      <div className="container mx-auto px-6">
        {/* Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-6">Our Vision for the Future</h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Building the world's most comprehensive circular economy platform, 
            powered by cutting-edge technology and global collaboration.
          </p>
        </div>

        {/* Vision Areas */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-20">
          {visionAreas.map((area, index) => (
            <div key={index} className="bg-slate-800 rounded-xl p-6 text-center">
              <div className="bg-emerald-500/10 p-4 rounded-full w-fit mx-auto mb-4">
                <area.icon className="h-8 w-8 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">{area.title}</h3>
              <p className="text-gray-300 mb-4">{area.description}</p>
              <div className="space-y-1">
                {area.metrics.map((metric, idx) => (
                  <div key={idx} className="text-sm text-emerald-400 font-medium">
                    {metric}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Roadmap */}
        <div className="mb-16">
          <h3 className="text-3xl font-bold text-white text-center mb-12">Development Roadmap</h3>
          <div className="space-y-8">
            {roadmapItems.map((item, index) => (
              <div key={index} className="flex items-start space-x-6">
                <div className="flex-shrink-0">
                  <div className={`p-3 rounded-full ${
                    item.status === 'in-progress' ? 'bg-emerald-500' :
                    item.status === 'planned' ? 'bg-blue-500' :
                    'bg-gray-500'
                  }`}>
                    <item.icon className="h-6 w-6 text-white" />
                  </div>
                </div>
                <div className="flex-1">
                  <div className="bg-slate-800 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <span className="text-emerald-400 font-semibold">{item.phase}</span>
                        <h4 className="text-xl font-bold text-white mt-1">{item.title}</h4>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        item.status === 'in-progress' ? 'bg-emerald-100 text-emerald-800' :
                        item.status === 'planned' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {item.status.replace('-', ' ')}
                      </span>
                    </div>
                    <p className="text-gray-300 mb-4">{item.description}</p>
                    <div className="grid md:grid-cols-2 gap-2">
                      {item.features.map((feature, idx) => (
                        <div key={idx} className="flex items-center space-x-2">
                          <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full"></div>
                          <span className="text-sm text-gray-400">{feature}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Call to Action */}
        <div className="bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-2xl p-12 text-center">
          <Rocket className="h-16 w-16 text-white mx-auto mb-6" />
          <h3 className="text-3xl font-bold text-white mb-4">
            Be Part of the Future
          </h3>
          <p className="text-white/90 mb-8 max-w-2xl mx-auto">
            Join us in building the world's most advanced circular economy platform. 
            Together, we can create a sustainable future for generations to come.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-white text-emerald-600 px-8 py-4 rounded-lg hover:bg-gray-100 transition font-semibold">
              Join Our Beta Program
            </button>
            <button className="border-2 border-white text-white px-8 py-4 rounded-lg hover:bg-white hover:text-emerald-600 transition font-semibold">
              Partner With Us
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}