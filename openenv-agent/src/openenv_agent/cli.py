"""CLI for openenv-agent."""

import sys
import logging
from typing import Optional

import click

from openenv_agent.client import OpenEnvClient
from openenv_agent.moderation_agent import ModerationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """OpenEnv Agent - Connect RL agents to OpenEnv servers."""
    pass


@main.command()
@click.argument("url")
@click.option("--max-steps", type=int, default=None, help="Max steps to run")
def run(url: str, max_steps: Optional[int]):
    """Run the moderation agent against an OpenEnv server."""
    client = OpenEnvClient(base_url=url)
    agent = ModerationAgent()

    env_name = getattr(agent, "model_name", "unknown")
    click.echo(f"[START] task=agent-run env={env_name} model=gpt-4.1")

    try:
        obs = client.reset()

        step = 0
        total_reward = 0.0
        last_error = None

        while True:
            if max_steps and step >= max_steps:
                break

            try:
                action = agent.predict(obs)
                next_obs, reward, done, info = client.step(action)
                total_reward += reward
                step += 1

                click.echo(
                    f"[STEP] step={step} action={action['decision']} "
                    f"reward={reward:.2f} done={done} error={last_error}"
                )

                if done:
                    break

                obs = next_obs
            except Exception as e:
                last_error = str(e)
                click.echo(
                    f"[STEP] step={step + 1} action=ALLOW "
                    f"reward=0.00 done=False error={last_error}"
                )
                step += 1
                last_error = None

        avg_reward = total_reward / max(step, 1)
        avg_reward = max(0.0, min(1.0, avg_reward))
        success = avg_reward >= 0.5
        click.echo(f"[END] success={success} steps={step} score={avg_reward:.4f}")

    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@main.command()
@click.argument("url")
def interactive(url: str):
    """Run the agent in interactive debug mode."""
    client = OpenEnvClient(base_url=url)
    agent = ModerationAgent()

    click.echo(f"Interactive mode - connecting to {url}")
    click.echo("Press Ctrl+C to exit\n")

    try:
        obs = client.reset()
        click.echo(f"Initial observation:\n{obs}\n")

        step = 0
        while True:
            click.echo(f"--- Step {step} ---")
            action = agent.predict(obs)
            click.echo(f"Action: {action}")

            next_obs, reward, done, info = client.step(action)
            click.echo(f"Reward: {reward:.2f}, Done: {done}")

            if done:
                click.echo("Episode complete!")
                break

            if click.confirm("Continue?"):
                obs = next_obs
                step += 1
            else:
                break

    except KeyboardInterrupt:
        click.echo("\nExiting...")
    finally:
        client.close()


@main.command()
@click.argument("url")
@click.option("--dataset", type=click.Path(), help="Path to dataset JSON")
def eval(url: str, dataset: Optional[str]):
    """Evaluate the agent on a dataset."""
    client = OpenEnvClient(base_url=url)
    agent = ModerationAgent()

    click.echo(f"[EVAL] Starting evaluation against {url}")

    total_reward = 0.0
    total_steps = 0
    episodes = 0

    try:
        while True:
            obs = client.reset()
            step = 0
            episode_reward = 0.0

            while True:
                action = agent.predict(obs)
                next_obs, reward, done, info = client.step(action)
                episode_reward += reward
                step += 1

                if done:
                    break

                obs = next_obs

            total_reward += episode_reward
            total_steps += step
            episodes += 1

            click.echo(f"[EPISODE] {episodes}: steps={step} reward={episode_reward:.2f}")

            # In real eval, would check against ground truth here
            if dataset:
                # TODO: evaluate against dataset
                pass

            if episodes >= 10:  # Eval on 10 episodes by default
                break

        click.echo(f"\n[EVAL COMPLETE]")
        click.echo(f"Episodes: {episodes}")
        click.echo(f"Total steps: {total_steps}")
        click.echo(f"Average reward: {total_reward / episodes:.4f}")

    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()