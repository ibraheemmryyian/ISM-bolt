import React from 'react';
import { Beaker, Building2, Recycle, LineChart, Users, Lightbulb } from 'lucide-react';

export function RoleInfo() {
  const roles = [
    {
      title: 'Researchers',
      icon: Beaker,
      description: 'Access industrial data for circular economy research, collaborate with industry partners, and develop innovative solutions.',
      benefits: [
        'Access to real-world industrial data',
        'Industry collaboration opportunities',
        'Research funding connections',
        'Publication opportunities'
      ]
    },
    {
      title: 'Industry Partners',
      icon: Building2,
      description: 'Transform waste into value, reduce disposal costs, and find sustainable material sources through our network.',
      benefits: [
        'Reduce waste management costs',
        'Find sustainable material sources',
        'Access circular economy expertise',
        'ESG reporting support'
      ]
    },
    {
      title: 'Waste Managers',
      icon: Recycle,
      description: 'Connect with industries to optimize waste collection, processing, and redistribution in the circular economy.',
      benefits: [
        'Optimize collection routes',
        'Find material buyers',
        'Track material flows',
        'Regulatory compliance support'
      ]
    },
    {
      title: 'Investors',
      icon: LineChart,
      description: 'Discover and support innovative circular economy projects and companies making real environmental impact.',
      benefits: [
        'Access vetted opportunities',
        'Impact measurement',
        'Industry insights',
        'Network connections'
      ]
    },
    {
      title: 'Consultants',
      icon: Users,
      description: 'Help organizations transition to circular business models and improve their sustainability performance.',
      benefits: [
        'Client matching',
        'Data-driven insights',
        'Best practice library',
        'Professional network'
      ]
    },
    {
      title: 'Innovators',
      icon: Lightbulb,
      description: 'Develop and scale new technologies and solutions for the circular economy.',
      benefits: [
        'Market validation',
        'Pilot opportunities',
        'Technical support',
        'Funding connections'
      ]
    }
  ];

  return (
    <div className="bg-slate-800 py-24">
      <div className="container mx-auto px-6">
        <h2 className="text-3xl font-bold text-white text-center mb-16">How You Can Help</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {roles.map((role, index) => (
            <div key={index} className="bg-slate-700 rounded-xl p-6">
              <div className="bg-emerald-500/10 p-3 rounded-lg w-fit mb-6">
                <role.icon className="h-6 w-6 text-emerald-400" />
              </div>
              
              <h3 className="text-xl font-semibold text-white mb-4">{role.title}</h3>
              
              <p className="text-gray-300 mb-6">{role.description}</p>
              
              <ul className="space-y-2">
                {role.benefits.map((benefit, idx) => (
                  <li key={idx} className="text-sm text-gray-400 flex items-center">
                    <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full mr-2"></span>
                    {benefit}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}